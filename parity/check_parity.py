"""Parity verification: `lilim.core` against `legacy/AuditedCode_1.py`.

This is the load-bearing claim of the whole `lilim/` deliverable. A rewrite
that silently diverges from the audited reference implementation is worse than
no rewrite, so this harness runs identical scenarios through both codebases and
compares by EXACT equality -- `np.array_equal` and `==`, not `allclose` --
everywhere except the one documented exception in Stage 5.

Run it:

    python lilim/parity/check_parity.py                # stages 0-3 and 5
    python lilim/parity/check_parity.py --with-oat     # adds stage 4 (slow)
    python lilim/parity/check_parity.py --report FILE  # write a markdown report

Stages
------
0  grid generation at seed 42            exact array equality on every field
1  study-wide reference cutoffs          exact equality on (Q1, Q2, Q3)
2  one seeded optimizer restart          score, placement, cooling field,
                                         convergence history; and proof that
                                         the iteration callback is inert
3  the full k sweep, 60 seeded restarts  against live legacy AND against the
                                         committed regeneration output
4  the 40-factor OAT sweep  (--with-oat) every sensitivity index, plus proof
                                         that shared species state was restored
5  coarse equity-weight map              DIVERGES BY DESIGN -- see below

The Stage 5 exception
---------------------
`get_coarse_cell_weights` in the reference implementation indexes a
fine-resolution array row-major when that array is ordered x-major, so the
coarse equity-weight map it returns is exactly its own transpose. `lilim` uses
the correct index. This is the ONLY permitted divergence; Stage 5 asserts the
relationship holds in the direction expected and validates `lilim`'s output
against a ground truth computed from coordinates, without index arithmetic at
all -- so the fix is not merely self-consistent.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LILIM = os.path.dirname(HERE)
REPO = os.path.dirname(LILIM)
if LILIM not in sys.path:
    sys.path.insert(0, LILIM)

from core import config                                          # noqa: E402
from core.aco import AntColonySystemACO                          # noqa: E402
from core.cooling import CoolingModel                            # noqa: E402
from core.grid import build_grid, without_equity_arm             # noqa: E402
from core.secpi import calibrate_reference_cutoffs               # noqa: E402
from core.sensitivity import SensitivityAnalyzer                 # noqa: E402
from core.species import TreeSpecies                             # noqa: E402

LEGACY_PATH = os.path.join(REPO, "legacy", "AuditedCode_1.py")


def _latest_headline() -> str | None:
    """The newest `results/run_*/data/headline.json`, or None if there is none.

    Discovered rather than hardcoded. A hardcoded path went stale the first time
    the pipeline was re-run (the crown-competition correction superseded the run
    this used to name -- see docs/DECISIONS.md D-21), and a stale committed
    reference is worse than no reference at all:
    the k-sweep would be compared against numbers a superseded model produced
    and would fail for the right reason with the wrong explanation. Run
    directories are named `run_<UTC timestamp>_<tag>`, so lexical order is
    chronological order.
    """
    pattern = os.path.join(REPO, "results", "run_*", "data", "headline.json")
    found = sorted(glob.glob(pattern))
    return found[-1] if found else None


HEADLINE_JSON = _latest_headline()

GRID_SEED = config.DEFAULT_GRID_SEED


# --------------------------------------------------------------------- utils
def load_legacy(path: str = LEGACY_PATH):
    """Load the reference implementation read-only, swallowing its prints."""
    spec = importlib.util.spec_from_file_location("secpi_reference", path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class Report:
    """Collects check outcomes so the summary is data, not printed side effects."""

    def __init__(self):
        self.checks: list[dict] = []
        self.stages: list[dict] = []

    def check(self, stage, name, ok, detail="", expected_divergence=False):
        self.checks.append({
            "stage": stage, "name": name, "ok": bool(ok),
            "detail": detail, "expected_divergence": expected_divergence,
        })
        mark = "OK  " if ok else "FAIL"
        if expected_divergence and ok:
            mark = "DIFF"
        print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
        return ok

    def stage(self, number, title, seconds):
        failures = [c for c in self.checks
                    if c["stage"] == number and not c["ok"]]
        self.stages.append({
            "stage": number, "title": title, "seconds": seconds,
            "n_checks": len([c for c in self.checks if c["stage"] == number]),
            "n_failures": len(failures),
        })

    @property
    def failures(self):
        return [c for c in self.checks if not c["ok"]]

    @property
    def passed(self):
        return not self.failures


def arrays_equal(a, b) -> tuple[bool, str]:
    """Exact equality, with a max-absolute-difference note when it fails."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if np.array_equal(a, b):
        return True, f"{a.size} values identical"
    try:
        diff = float(np.max(np.abs(a.astype(float) - b.astype(float))))
    except (TypeError, ValueError):
        return False, "unequal (non-numeric)"
    n_diff = int(np.sum(a != b))
    return False, f"{n_diff}/{a.size} differ, max abs {diff:.3e}"


# ------------------------------------------------------------------- stage 0
def stage0_grid(legacy, rep: Report):
    print("\nStage 0 - grid generation at seed %d" % GRID_SEED)
    t0 = time.time()

    with quiet():
        np.random.seed(GRID_SEED)
        lg = legacy.TwoLevelUrbanGrid(
            coarse_width=config.COARSE_GRID["width"],
            coarse_height=config.COARSE_GRID["height"],
            coarse_cell_size=config.COARSE_GRID["cell_size"],
            fine_cell_size=config.FINE_GRID["cell_size"],
        )
        lg.generate_ca_archetype(params=dict(config.CA_PARAMS),
                                 morphology=config.CA_PARAMS["morphology"])

    ng = build_grid(grid_seed=GRID_SEED)

    for field in ("coarse_grid", "fine_grid", "plantable_coords",
                  "vulnerable_coords", "vulnerability_weights"):
        ok, detail = arrays_equal(getattr(lg, field), getattr(ng, field))
        rep.check(0, f"grid.{field}", ok, detail)

    comp = ng.composition()
    rep.check(0, "land-use composition in target bands",
              55 <= comp["pct_p"] + comp["pct_v"] <= 75 and 5 <= comp["pct_v"] <= 10,
              f"P={comp['n_p']} A={comp['n_a']} V={comp['n_v']} "
              f"plantable={comp['n_plantable']}")

    rep.stage(0, "grid generation", time.time() - t0)
    return lg, ng


# ------------------------------------------------------------------- stage 1
def stage1_cutoffs(legacy, lg, ng, rep: Report):
    print("\nStage 1 - study-wide reference cutoffs")
    t0 = time.time()

    with quiet():
        legacy_model = legacy.CorrectedCoolingModel(
            decay_lambda=config.COOLING_PARAMS["decay_lambda"],
            cca_threshold=config.COOLING_PARAMS["cca_threshold"],
            competition_k=config.COOLING_PARAMS["competition_k"],
        )
        legacy_cutoffs = legacy.calibrate_global_reference_cutoffs(
            lg, legacy_model, legacy_model.tree_species.species_list,
            n_trees_range=config.CUTOFF_CALIB["n_trees_range"],
            n_samples=config.CUTOFF_CALIB["n_samples"],
            random_seed=config.CUTOFF_CALIB["random_seed"],
        )

    new_model = CoolingModel()
    new_cutoffs = calibrate_reference_cutoffs(
        ng, new_model, new_model.tree_species.species_list)

    ok = all(a == b for a, b in zip(legacy_cutoffs, new_cutoffs))
    rep.check(1, "reference cutoffs (Q1, Q2, Q3)", ok,
              "(%.12f, %.12f, %.12f)" % tuple(new_cutoffs) if ok
              else f"legacy {legacy_cutoffs} vs lilim {new_cutoffs}")

    rep.stage(1, "reference cutoffs", time.time() - t0)
    return legacy_model, legacy_cutoffs, new_model, new_cutoffs


# ------------------------------------------------------------------- stage 2
def stage2_single_run(legacy, lg, ng, legacy_model, legacy_cutoffs,
                      new_model, new_cutoffs, rep: Report):
    print("\nStage 2 - one seeded optimizer restart")
    t0 = time.time()

    seed = config.kseed(5, 0)
    p = config.ACO_PARAMS

    with quiet():
        laco = legacy.AntColonySystemACO(
            lg, legacy_model, n_trees=p["n_trees"], n_ants=p["n_ants"],
            n_iterations=p["n_iterations"], evaporation_rate=p["evaporation_rate"],
            alpha=p["alpha"], beta=p["beta"], q0=p["q0"],
            random_seed=seed, reference_cutoffs=legacy_cutoffs,
        )
        lhb, lha = laco.run(verbose=False)
    legacy_seconds = time.time() - t0

    naco = AntColonySystemACO(
        ng, new_model, n_trees=p["n_trees"], random_seed=seed,
        reference_cutoffs=new_cutoffs)
    nhb, nha = naco.run()

    rep.check(2, "best raw SECPI", laco.best_secpi == naco.best_secpi,
              f"{naco.best_secpi!r} (seed {seed})")

    lcoords, lspecies = laco.best_solution
    ncoords, nspecies = naco.best_solution
    ok, detail = arrays_equal(np.asarray(lcoords), np.asarray(ncoords))
    rep.check(2, "best-solution coordinates", ok, detail)
    rep.check(2, "best-solution species", list(lspecies) == list(nspecies),
              " + ".join(nspecies))

    ok, detail = arrays_equal(laco.best_cooling, naco.best_cooling)
    rep.check(2, "best cooling field", ok, detail)

    ok, detail = arrays_equal(np.asarray(lhb), np.asarray(nhb))
    rep.check(2, "history_best per iteration", ok, detail)
    ok, detail = arrays_equal(np.asarray(lha), np.asarray(nha))
    rep.check(2, "history_avg per iteration", ok, detail)

    # The callback must be inert: same seed, same answer, with a callback that
    # does real work but consumes no randomness.
    seen: list[tuple] = []
    caco = AntColonySystemACO(
        ng, new_model, n_trees=p["n_trees"], random_seed=seed,
        reference_cutoffs=new_cutoffs)
    chb, cha = caco.run(on_iteration=lambda i, b, a, hb, ha: seen.append((i, b, a)))

    rep.check(2, "iteration callback is inert (score unchanged)",
              caco.best_secpi == naco.best_secpi, f"{len(seen)} callbacks fired")
    ok, _ = arrays_equal(np.asarray(chb), np.asarray(nhb))
    rep.check(2, "iteration callback is inert (history unchanged)", ok)
    rep.check(2, "callback saw every iteration",
              len(seen) == p["n_iterations"],
              f"{len(seen)} of {p['n_iterations']}")

    print(f"       (one restart ~ {legacy_seconds:.1f}s)")
    rep.stage(2, "single seeded restart", time.time() - t0)
    return legacy_seconds


# ------------------------------------------------------------------- stage 3
def stage3_k_sweep(legacy, lg, ng, legacy_model, legacy_cutoffs,
                   new_model, new_cutoffs, rep: Report):
    print("\nStage 3 - full k sweep, 60 seeded restarts, both arms")
    t0 = time.time()

    with quiet():
        lg_no = legacy.TwoLevelUrbanGrid(
            coarse_width=lg.coarse_width, coarse_height=lg.coarse_height,
            coarse_cell_size=lg.coarse_cell_size, fine_cell_size=lg.fine_cell_size)
        lg_no.coarse_grid = lg.coarse_grid.copy()
        lg_no.fine_grid = lg.fine_grid.copy()
        lg_no.plantable_coords = lg.plantable_coords.copy()
        lg_no.vulnerable_coords = (lg.vulnerable_coords.copy()
                                   if len(lg.vulnerable_coords) > 0 else np.array([]))
        lg_no.vulnerability_weights = lg.vulnerability_weights.copy()
        lg_no.convert_vulnerable_to_prohibited()

    ng_no = without_equity_arm(ng)

    ok, detail = arrays_equal(lg_no.vulnerability_weights, ng_no.vulnerability_weights)
    rep.check(3, "WITHOUT-arm equity weights", ok, detail)
    ok, detail = arrays_equal(lg_no.coarse_grid, ng_no.coarse_grid)
    rep.check(3, "WITHOUT-arm coarse grid", ok, detail)

    p = config.ACO_PARAMS
    arms = {"WITH": (lg, ng), "WITHOUT": (lg_no, ng_no)}
    live = {"WITH": {}, "WITHOUT": {}}
    mismatches = []

    for arm, (larm, narm) in arms.items():
        for k in config.K_VALUES:
            legacy_vals, new_vals = [], []
            for run in range(config.N_RUNS):
                seed = config.kseed(k, run)
                with quiet():
                    laco = legacy.AntColonySystemACO(
                        larm, legacy_model, n_trees=k, n_ants=p["n_ants"],
                        n_iterations=p["n_iterations"],
                        evaporation_rate=p["evaporation_rate"],
                        alpha=p["alpha"], beta=p["beta"], q0=p["q0"],
                        random_seed=seed, reference_cutoffs=legacy_cutoffs)
                    laco.run(verbose=False)
                naco = AntColonySystemACO(
                    narm, new_model, n_trees=k, random_seed=seed,
                    reference_cutoffs=new_cutoffs)
                naco.run()

                legacy_vals.append(float(laco.best_secpi))
                new_vals.append(float(naco.best_secpi))
                if laco.best_secpi != naco.best_secpi:
                    mismatches.append((arm, k, run, laco.best_secpi, naco.best_secpi))
            live[arm][k] = new_vals
            print(f"       {arm:8s} k={k}  " +
                  "  ".join(f"{v:.6f}" for v in new_vals))

    rep.check(3, "60 restarts match live legacy exactly", not mismatches,
              "all 60 identical" if not mismatches else f"{len(mismatches)} differ")

    # The stronger check: against the committed regeneration output, which is
    # where the reported numbers actually come from.
    if HEADLINE_JSON and os.path.exists(HEADLINE_JSON):
        print(f"    regeneration reference: "
              f"{os.path.relpath(HEADLINE_JSON, REPO)}")
        with open(HEADLINE_JSON) as f:
            headline = json.load(f)
        arch_mismatch = []
        n_compared = 0
        for arm in ("WITH", "WITHOUT"):
            for row in headline["ksweep"][arm]:
                k = row["k"]
                archived = row["individual_values"]
                produced = live[arm][k]
                for i, (a, b) in enumerate(zip(archived, produced)):
                    n_compared += 1
                    if a != b:
                        arch_mismatch.append((arm, k, i, a, b))
        rep.check(3, "60 restarts match the committed regeneration output",
                  not arch_mismatch,
                  f"{n_compared} values from headline.json"
                  if not arch_mismatch else f"{len(arch_mismatch)} differ")
    else:
        rep.check(3, "committed regeneration output available", False,
                  "no results/run_*/data/headline.json on disk")

    rep.stage(3, "k sweep", time.time() - t0)
    return live


# ------------------------------------------------------------------- stage 4
def stage4_oat(legacy, lg, ng, legacy_model, legacy_cutoffs,
               new_model, new_cutoffs, rep: Report, n_samples=1):
    print(f"\nStage 4 - OAT sensitivity sweep, n_samples={n_samples}")
    t0 = time.time()

    before = {k: dict(v) for k, v in TreeSpecies.SPECIES_DATA.items()}

    with quiet():
        lan = legacy.SensitivityAnalyzer(
            lg, legacy_model, dict(config.ACO_PARAMS), HERE,
            reference_cutoffs=legacy_cutoffs)
        np.random.seed(GRID_SEED)
        legacy_rows = lan.run_oat_analysis(n_samples=n_samples)

    legacy_by_name = {r["parameter"]: r for r in lan.results}

    nan = SensitivityAnalyzer(ng, dict(config.ACO_PARAMS),
                              reference_cutoffs=new_cutoffs)
    np.random.seed(GRID_SEED)
    new_rows = nan.run_oat_analysis(n_samples=n_samples)

    rep.check(4, "factor count", len(new_rows) == len(legacy_by_name) == 40,
              f"{len(new_rows)} factors")
    rep.check(4, "baseline SECPI",
              lan.baseline_secpi == nan.baseline_secpi,
              f"{nan.baseline_secpi!r}")

    diffs = []
    for row in new_rows:
        lrow = legacy_by_name.get(row["parameter"])
        if lrow is None:
            diffs.append((row["parameter"], "missing in legacy", ""))
            continue
        for field in ("secpi_low", "secpi_high", "sensitivity_index"):
            if lrow[field] != row[field]:
                diffs.append((row["parameter"], field,
                              f"{lrow[field]!r} vs {row[field]!r}"))
    rep.check(4, "all 40 sensitivity indices match", not diffs,
              "identical" if not diffs else f"{len(diffs)} field differences")

    after = {k: dict(v) for k, v in TreeSpecies.SPECIES_DATA.items()}
    restored = all(
        before[s][f] == after[s][f]
        for s in before for f in before[s] if f != "CPA")
    rep.check(4, "shared species state restored after sweep", restored,
              "SPECIES_DATA unchanged")

    rep.stage(4, "OAT sensitivity", time.time() - t0)
    return legacy_rows, new_rows


# ------------------------------------------------------------------- stage 5
def stage5_equity_map(lg, ng, rep: Report):
    print("\nStage 5 - coarse equity-weight map (the one documented divergence)")
    t0 = time.time()

    legacy_cw = lg.get_coarse_cell_weights()
    new_cw = ng.get_coarse_cell_weights()
    truth = ng.coarse_cell_weights_by_coordinate()

    ok_truth, detail = arrays_equal(new_cw, truth)
    rep.check(5, "lilim map matches coordinate-derived ground truth",
              ok_truth, detail)

    is_transpose = np.allclose(legacy_cw, new_cw.T)
    differs = not np.allclose(legacy_cw, new_cw)
    max_abs = float(np.max(np.abs(legacy_cw - new_cw)))

    rep.check(5, "legacy map is the transpose of the corrected one",
              is_transpose and differs,
              f"max abs difference {max_abs:.6f}", expected_divergence=True)

    rep.check(5, "divergence is confined to this function",
              np.array_equal(lg.vulnerability_weights, ng.vulnerability_weights),
              "fine-resolution equity weights identical in both")

    rep.stage(5, "equity-weight map", time.time() - t0)
    return legacy_cw, new_cw, max_abs


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-oat", action="store_true",
                    help="also run stage 4 (the 40-factor sweep; slow)")
    ap.add_argument("--oat-samples", type=int, default=1,
                    help="n_samples for stage 4 (default 1; production is 3)")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip stage 3 (the 60-restart k sweep)")
    ap.add_argument("--report", default=None,
                    help="write a markdown report to this path")
    args = ap.parse_args()

    print("=" * 72)
    print("lilim parity check  vs  legacy/AuditedCode_1.py")
    print("=" * 72)
    print(f"reference : {LEGACY_PATH}")
    print(f"numpy     : {np.__version__}")
    print(f"grid seed : {GRID_SEED}")

    started = time.time()
    rep = Report()
    legacy = load_legacy()

    lg, ng = stage0_grid(legacy, rep)
    legacy_model, legacy_cutoffs, new_model, new_cutoffs = stage1_cutoffs(
        legacy, lg, ng, rep)
    stage2_single_run(legacy, lg, ng, legacy_model, legacy_cutoffs,
                      new_model, new_cutoffs, rep)

    if not args.skip_sweep:
        stage3_k_sweep(legacy, lg, ng, legacy_model, legacy_cutoffs,
                       new_model, new_cutoffs, rep)

    if args.with_oat:
        stage4_oat(legacy, lg, ng, legacy_model, legacy_cutoffs,
                   new_model, new_cutoffs, rep, n_samples=args.oat_samples)

    stage5_equity_map(lg, ng, rep)

    elapsed = time.time() - started
    print("\n" + "=" * 72)
    for s in rep.stages:
        status = "PASS" if s["n_failures"] == 0 else f"{s['n_failures']} FAILED"
        print(f"Stage {s['stage']}  {s['title']:<28s} {s['n_checks']:>2d} checks  "
              f"{status:>10s}  {s['seconds']:>6.1f}s")
    print("=" * 72)
    print(f"{len(rep.checks)} checks, {len(rep.failures)} failures, {elapsed:.1f}s total")

    if rep.failures:
        print("\nFAILURES:")
        for c in rep.failures:
            print(f"  stage {c['stage']}: {c['name']} - {c['detail']}")

    if args.report:
        write_report(args.report, rep, elapsed)
        print(f"\nreport written: {args.report}")

    return 0 if rep.passed else 1


def write_report(path, rep: Report, elapsed):
    import datetime
    lines = [
        "# Parity report — `lilim/core` against `legacy/AuditedCode_1.py`",
        "",
        f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"numpy {np.__version__} · grid seed {GRID_SEED} · {elapsed:.1f}s",
        "",
        "Comparison is by exact equality (`np.array_equal`, `==`), not tolerance,",
        "except for the single documented divergence in Stage 5.",
        "",
        "| Stage | Checks | Failures | Seconds |",
        "|---|---|---|---|",
    ]
    for s in rep.stages:
        lines.append(f"| {s['stage']} — {s['title']} | {s['n_checks']} | "
                     f"{s['n_failures']} | {s['seconds']:.1f} |")
    lines += ["", "## Every check", "", "| Stage | Check | Result | Detail |", "|---|---|---|---|"]
    for c in rep.checks:
        result = "pass"
        if not c["ok"]:
            result = "**FAIL**"
        elif c["expected_divergence"]:
            result = "diverges by design"
        lines.append(f"| {c['stage']} | {c['name']} | {result} | {c['detail']} |")
    lines += ["", f"**{len(rep.checks)} checks, {len(rep.failures)} failures.**", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
