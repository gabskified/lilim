"""Scenario sweeps: tree count k, species palette, and the equity contrast.

Consolidates the sweep logic from `legacy/AuditedCode_1.py`'s
`SuboptimalScenariosGenerator` (:2799-3226) with the seeding and
outcome-metric discipline from `tools/secpi_results/secpi_results_lib.py`.

WHY THE SWEEP LOGIC COMES FROM `tools/`, NOT FROM THE MONOLITH
--------------------------------------------------------------
`SuboptimalScenariosGenerator.run_optimization_for_k` keeps only the best of
its five restarts -- the other four placements and cooling fields are thrown
away and only their scalar scores survive. It also never passes a seed, so its
restarts cannot be reproduced. Both are fixed here: every restart is seeded
explicitly, and every restart's placement and cooling field is retained.

THE TWO ARMS
------------
WITH    -- the base grid, equity weights as generated.
WITHOUT -- the same grid with Vulnerable cells converted to Prohibited, which
           empties `vulnerable_coords` and flattens every equity weight to 1.0.

Both arms' outcome metrics are computed against the SHARED BASE-GRID Vulnerable
mask. This is not a convenience. Measuring the WITHOUT arm against its own
V-state would return zero by construction and the contrast would be vacuous.

The same seed is used for restart index `run` in both arms at the same k, so a
pair differs only by the equity manipulation and not also by its pheromone
initialisation.
"""
from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np

from . import config, secpi as secpi_mod
from .aco import run_once
from .grid import without_equity_arm


def species_subsets(species_list: Sequence[str]) -> list[dict]:
    """All non-empty subsets of the palette, ordered by size then by combination.

    For the six-species palette this is 63 subsets. `s` is the palette size --
    the sweep axis for the diversity analysis, distinct from `k`, the number of
    trees planted.
    """
    out = []
    sid = 0
    for s in range(1, len(species_list) + 1):
        for combo in itertools.combinations(species_list, s):
            sid += 1
            out.append({"subset_id": sid, "s": s, "species_available": list(combo)})
    return out


def summarise(values: Sequence[float]) -> dict:
    """n / mean / sd / cv / min / max for a set of restart scores."""
    a = np.asarray(list(values), dtype=float)
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "cv": None, "min": None, "max": None}
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "cv": float(sd / mean) if mean != 0.0 else None,
        "min": float(a.min()),
        "max": float(a.max()),
        "individual_values": [float(v) for v in a],
    }


class ScenarioContext:
    """Both arms plus the shared masks, built once and reused by every sweep."""

    def __init__(self, grid, cooling_model, cutoffs):
        self.grid = grid
        self.grid_no_equity = without_equity_arm(grid)
        self.cooling_model = cooling_model
        self.cutoffs = cutoffs
        self.v_fine_mask = secpi_mod.vulnerable_fine_mask(grid)
        self.n_v_fine = int(self.v_fine_mask.sum())
        adj = secpi_mod.adjacency_masks(grid)
        self.adj8 = adj["adj8"]
        self.adj4 = adj["adj4"]
        self.cell_size = float(grid.coarse_cell_size)

    def arm(self, name: str):
        return self.grid if name == "WITH" else self.grid_no_equity


def run_restart(ctx: ScenarioContext, arm: str, n_trees: int, seed: int,
                species_subset=None, aco_params=None) -> dict:
    """One seeded restart in one arm, with its outcome metrics attached."""
    grid = ctx.arm(arm)
    aco, res = run_once(
        grid, ctx.cooling_model, ctx.cutoffs,
        n_trees=n_trees, species_subset=species_subset,
        seed=seed, aco_params=aco_params,
    )
    cooling = np.asarray(aco.best_cooling, dtype=float).reshape(-1)
    coords = aco.best_solution[0]

    res.update({
        "arm": arm,
        # Both metrics use the BASE-grid V mask in both arms -- see the module
        # docstring for why.
        "cooling_share_in_vulnerable": secpi_mod.cooling_share_in_vulnerable(
            cooling, ctx.v_fine_mask),
        "trees_adjacent_share_8conn": secpi_mod.trees_adjacent_share(
            coords, ctx.adj8, ctx.cell_size),
        "trees_adjacent_share_4conn": secpi_mod.trees_adjacent_share(
            coords, ctx.adj4, ctx.cell_size),
    })
    return res


def run_k_sweep(ctx: ScenarioContext, k_values=None, n_runs=None,
                arms=("WITH", "WITHOUT"), aco_params=None,
                progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Tree-count sweep: k = 1..6 x both arms x n_runs seeded restarts.

    `progress(done, total, label)` is called after each restart so a UI can
    show real progress rather than a spinner. It must not consume randomness.
    """
    k_values = k_values or config.K_VALUES
    n_runs = n_runs if n_runs is not None else config.N_RUNS

    total = len(k_values) * len(arms) * n_runs
    done = 0
    restarts: list[dict] = []

    for arm in arms:
        for k in k_values:
            for run in range(n_runs):
                seed = config.kseed(k, run)
                res = run_restart(ctx, arm, n_trees=k, seed=seed,
                                  aco_params=aco_params)
                res.update({"block": "ksweep", "k": k, "run": run})
                restarts.append(res)
                done += 1
                if progress:
                    progress(done, total, f"{arm} k={k} restart {run + 1}/{n_runs}")

    return {"restarts": restarts, "summary": summarise_k_sweep(restarts, k_values, arms)}


def summarise_k_sweep(restarts, k_values=None, arms=("WITH", "WITHOUT")) -> dict:
    """Per-arm, per-k summaries plus the arm contrast at each k."""
    k_values = k_values or config.K_VALUES
    out: dict = {arm: [] for arm in arms}

    for arm in arms:
        for k in k_values:
            rows = [r for r in restarts if r["arm"] == arm and r["k"] == k]
            stats = summarise([r["raw_secpi"] for r in rows])
            stats.update({
                "k": k,
                "cooling_share_in_vulnerable_mean": float(np.mean(
                    [r["cooling_share_in_vulnerable"] for r in rows])) if rows else None,
                "trees_adjacent_share_8conn_mean": float(np.mean(
                    [r["trees_adjacent_share_8conn"] for r in rows])) if rows else None,
                "mean_species_actually_used": float(np.mean(
                    [r["species_actually_used"] for r in rows])) if rows else None,
            })
            out[arm].append(stats)

    if set(arms) == {"WITH", "WITHOUT"}:
        contrast = []
        for k in k_values:
            w = next(s for s in out["WITH"] if s["k"] == k)
            wo = next(s for s in out["WITHOUT"] if s["k"] == k)
            if w["mean"] is None or wo["mean"] is None:
                continue
            diff = w["mean"] - wo["mean"]
            contrast.append({
                "k": k,
                "with_mean": w["mean"],
                "without_mean": wo["mean"],
                "difference": diff,
                "percent_of_with": 100.0 * diff / w["mean"] if w["mean"] else None,
            })
        out["arm_contrast_by_k"] = contrast

    return out


def run_subset_sweep(ctx: ScenarioContext, subsets=None, n_runs=None,
                     arms=("WITH", "WITHOUT"), n_trees=None, aco_params=None,
                     progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Species-palette sweep: every subset x both arms x n_runs restarts.

    Tree count is held fixed (five by default) so palette size `s` is the only
    axis moving. The full 63-subset sweep is 630 restarts and is a batch job,
    not an interactive one -- pass a reduced `subsets` list for exploration.
    """
    species_list = ctx.cooling_model.tree_species.species_list
    subsets = subsets if subsets is not None else species_subsets(species_list)
    n_runs = n_runs if n_runs is not None else config.N_RUNS
    n_trees = n_trees if n_trees is not None else config.SUBSET_N_TREES

    total = len(subsets) * len(arms) * n_runs
    done = 0
    restarts: list[dict] = []

    for sub in subsets:
        for arm in arms:
            for run in range(n_runs):
                seed = config.sseed(sub["subset_id"], run)
                res = run_restart(ctx, arm, n_trees=n_trees, seed=seed,
                                  species_subset=sub["species_available"],
                                  aco_params=aco_params)
                res.update({
                    "block": "subset",
                    "subset_id": sub["subset_id"],
                    "s": sub["s"],
                    "species_available": list(sub["species_available"]),
                    "run": run,
                })
                restarts.append(res)
                done += 1
                if progress:
                    progress(done, total,
                             f"{arm} subset {sub['subset_id']}/{len(subsets)} "
                             f"restart {run + 1}/{n_runs}")

    return {"restarts": restarts, "summary": summarise_subset_sweep(restarts, arms)}


def summarise_subset_sweep(restarts, arms=("WITH", "WITHOUT")) -> dict:
    """Per-subset best/mean, and the mean by palette size."""
    out: dict = {}
    for arm in arms:
        rows = [r for r in restarts if r["arm"] == arm]
        by_subset: dict = {}
        for r in rows:
            by_subset.setdefault(r["subset_id"], []).append(r)

        subset_stats = []
        for sid, group in sorted(by_subset.items()):
            scores = [g["raw_secpi"] for g in group]
            best = max(group, key=lambda g: g["raw_secpi"])
            subset_stats.append({
                "subset_id": sid,
                "s": group[0]["s"],
                "species_available": group[0]["species_available"],
                "best_raw_secpi": float(max(scores)),
                "secpi_mean": float(np.mean(scores)),
                "secpi_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "species_used_at_best": sorted(set(best["species"])),
                "species_actually_used": best["species_actually_used"],
                "all_available_used": int(
                    set(best["species"]) == set(group[0]["species_available"])),
            })
        subset_stats.sort(key=lambda d: d["best_raw_secpi"], reverse=True)
        for rank, row in enumerate(subset_stats, start=1):
            row["rank"] = rank

        by_s: dict = {}
        for row in subset_stats:
            by_s.setdefault(row["s"], []).append(row["secpi_mean"])
        out[arm] = {
            "subsets": subset_stats,
            "mean_by_palette_size": {s: float(np.mean(v)) for s, v in sorted(by_s.items())},
        }
    return out
