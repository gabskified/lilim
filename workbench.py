"""lilim — an interactive workbench over the SECPI reference implementation.

Run it:

    streamlit run lilim/workbench.py

WHAT THIS FILE IS AND IS NOT
----------------------------
This is presentation only. Every number shown here is computed in `core/`, and
every chart is drawn by `viz/figures.py`. Nothing scientific is decided in this
file, and nothing here is a second implementation of anything.

TWO STREAMLIT HAZARDS THIS FILE HANDLES DELIBERATELY
-----------------------------------------------------
1. Streamlit re-runs this entire script on every widget interaction. The
   reference implementation's determinism rests on GLOBAL numpy random state,
   so a re-run at the wrong moment could silently change results. Every seeded
   operation in `core/` sets its own seed immediately before use rather than
   relying on ambient state, and everything expensive is cached on its full
   parameter set. Neither is optional.

2. Widgets must never change a displayed number without the user asking. Long
   analyses are therefore explicit button presses that write into
   `st.session_state`, not reactive recomputations, and every stored result
   carries the settings it was produced under so a stale result can be
   labelled as stale instead of quietly misread.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core import config                                    # noqa: E402
from core import build_context                             # noqa: E402
from core.scenarios import ScenarioContext, run_k_sweep    # noqa: E402
from core.sensitivity import SensitivityAnalyzer           # noqa: E402
from core.secpi import BASELINE_NORMALIZED                 # noqa: E402
from viz import export as vx                               # noqa: E402
from viz import figures as fg                              # noqa: E402
from viz import theme                                      # noqa: E402

st.set_page_config(
    page_title="lilim — SECPI workbench",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.register()
st.markdown(f"<style>{theme.css()}</style>", unsafe_allow_html=True)

PLOTLY_CONFIG = {"displaylogo": False,
                 "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def note(html: str):
    st.markdown(f'<div class="lilim-note">{html}</div>', unsafe_allow_html=True)


def rule():
    st.markdown('<hr class="lilim-rule">', unsafe_allow_html=True)


def chart(fig, name: str | None = None, caption: str | None = None, **kwargs):
    """Render a figure, its caption, and an export control.

    The caption is page text rather than a Plotly annotation. That is
    deliberate and is the fix for a real bug: an in-figure caption is placed in
    paper coordinates that scale with the figure's own height, so against a
    fixed bottom margin it clipped or collided on almost every figure and
    drifted as the window resized. As flowing HTML it simply reflows.

    Pass `name` to get the caption (looked up in `figures.CAPTIONS`) and the
    save control. The export path re-attaches the caption to the figure, so a
    saved file still carries its caveat.
    """
    key = kwargs.pop("key", None) or (f"fig_{name}" if name else None)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key, **kwargs)

    if caption is None and name:
        caption = fg.CAPTIONS.get(name)
    if caption:
        st.markdown(f'<div class="lilim-caption">{caption}</div>',
                    unsafe_allow_html=True)
    if name:
        export_control(fig, name, caption)


@st.fragment
def export_control(fig, name: str, caption: str | None):
    """A compact save control, isolated in a fragment.

    The fragment matters: without it, pressing a save button reruns the entire
    script, re-rendering every tab. Rendering is also lazy -- bytes are only
    produced when asked for, never eagerly for sixteen figures on every rerun.
    """
    label = name.replace("_", " ")
    with st.popover("Save figure", use_container_width=False):
        st.caption(f"**{label}** — the caption above is included in the file.")

        ok, reason = vx.available()
        state_key = f"export_{name}"

        if ok:
            fmt = st.radio("Format", ["PNG (300 dpi)", "SVG (vector)",
                                      "HTML (interactive)"],
                           key=f"{state_key}_fmt", horizontal=False)
        else:
            fmt = "HTML (interactive)"
            st.caption(f"PNG and SVG unavailable — {reason}")

        if st.button("Prepare file", key=f"{state_key}_go"):
            try:
                if fmt.startswith("PNG"):
                    data, ext, mime = (vx.to_png_bytes(fig, caption), "png", "image/png")
                elif fmt.startswith("SVG"):
                    data, ext, mime = (vx.to_svg_bytes(fig, caption), "svg", "image/svg+xml")
                else:
                    data, ext, mime = (vx.to_html_bytes(fig, caption, label),
                                       "html", "text/html")
                st.session_state[state_key] = (data, vx.filename(name, ext), mime)
            except Exception as exc:
                st.session_state.pop(state_key, None)
                st.error(f"Could not render: {exc}")

        ready = st.session_state.get(state_key)
        if ready:
            data, fname, mime = ready
            st.download_button(f"Download  ({len(data) / 1024:.0f} KB)",
                               data=data, file_name=fname, mime=mime,
                               key=f"{state_key}_dl")


# ---------------------------------------------------------------- cached work
@st.cache_data(show_spinner="Generating grid and calibrating cutoffs…")
def cached_context(grid_seed, morphology, p_init, gamma, theta,
                   decay_lambda, cca_threshold, competition_k):
    """Grid + cooling model + study-wide cutoffs, keyed on everything that
    affects them. Returns the StudyContext; Streamlit caches it by value."""
    return build_context(
        grid_seed=grid_seed,
        ca_params={"p_init": p_init, "gamma": gamma, "theta": theta},
        morphology=morphology,
        cooling_params={"decay_lambda": decay_lambda,
                        "cca_threshold": cca_threshold,
                        "competition_k": competition_k},
    )


# ------------------------------------------------------------------- sidebar
def sidebar():
    with st.sidebar:
        st.markdown("## lilim")
        st.caption("Tagalog: the shade a tree casts.")

        if st.button("Reset to production values", width='stretch'):
            for key in list(st.session_state.keys()):
                if key.startswith("w_"):
                    del st.session_state[key]
            st.rerun()

        st.markdown("## Domain")
        grid_seed = st.number_input(
            "Grid seed", min_value=0, max_value=999_999,
            value=config.DEFAULT_GRID_SEED, step=1, key="w_seed",
            help="42 is the canonical grid behind every reported result.")
        morphology = st.selectbox(
            "Urban morphology", config.MORPHOLOGIES, index=0, key="w_morph",
            help="Organic clusters around dense neighbourhoods; linear favours "
                 "a central band.")

        with st.expander("Cellular automaton"):
            p_init = st.slider("Initial seed density", 0.05, 0.40,
                               config.CA_PARAMS["p_init"], 0.01, key="w_pinit")
            gamma = st.slider("Growth rate γ", 0.5, 8.0,
                              config.CA_PARAMS["gamma"], 0.1, key="w_gamma")
            theta = st.slider("Clustering threshold θ", 1, 6,
                              config.CA_PARAMS["theta"], 1, key="w_theta")
            st.caption("p₀ is fixed at 1.0. It and γ are both multiplicative "
                       "scale factors in the first update, so they are not "
                       "separately identifiable; γ alone carries the calibration.")

        st.markdown("## Cooling")
        decay_lambda = st.slider("Decay λ", 0.5, 3.0,
                                 config.COOLING_PARAMS["decay_lambda"], 0.1,
                                 key="w_lambda")
        cca_threshold = st.slider("Competition threshold", 0.5, 2.0,
                                  config.COOLING_PARAMS["cca_threshold"], 0.1,
                                  key="w_cca")
        competition_k = st.slider("Competition steepness K", 1.0, 10.0,
                                  config.COOLING_PARAMS["competition_k"], 0.5,
                                  key="w_k")

        st.markdown("## Optimizer")
        n_trees = st.slider("Trees to plant (k)", 1, 12,
                            config.ACO_PARAMS["n_trees"], 1, key="w_ntrees")
        n_ants = st.slider("Ants per iteration", 5, 50,
                           config.ACO_PARAMS["n_ants"], 5, key="w_ants")
        n_iterations = st.slider("Iterations", 5, 100,
                                 config.ACO_PARAMS["n_iterations"], 5, key="w_iters")
        q0 = st.slider("Exploitation rate q₀", 0.0, 1.0,
                       config.ACO_PARAMS["q0"], 0.05, key="w_q0",
                       help="Probability of taking the best available option "
                            "outright rather than sampling.")
        aco_seed = st.number_input(
            "Optimizer seed", min_value=0, max_value=9_999_999,
            value=config.kseed(config.ACO_PARAMS["n_trees"], 0), step=1,
            key="w_acoseed")

        with st.expander("Advanced optimizer"):
            evaporation = st.slider("Evaporation rate", 0.05, 0.95,
                                    config.ACO_PARAMS["evaporation_rate"], 0.05,
                                    key="w_evap")
            alpha = st.slider("Pheromone weight α", 0.0, 4.0,
                              config.ACO_PARAMS["alpha"], 0.1, key="w_alpha")
            beta = st.slider("Heuristic weight β", 0.0, 6.0,
                             config.ACO_PARAMS["beta"], 0.1, key="w_beta")

    return {
        "grid_seed": int(grid_seed), "morphology": morphology,
        "p_init": p_init, "gamma": gamma, "theta": int(theta),
        "decay_lambda": decay_lambda, "cca_threshold": cca_threshold,
        "competition_k": competition_k,
        "aco_params": {
            "n_trees": int(n_trees), "n_ants": int(n_ants),
            "n_iterations": int(n_iterations), "q0": q0,
            "evaporation_rate": evaporation, "alpha": alpha, "beta": beta,
        },
        "aco_seed": int(aco_seed),
    }


def production_badge(s):
    """Say plainly whether what is on screen is the reported configuration."""
    departures = []
    if s["grid_seed"] != config.DEFAULT_GRID_SEED:
        departures.append(f"grid seed {s['grid_seed']}")
    if s["morphology"] != config.CA_PARAMS["morphology"]:
        departures.append(f"morphology {s['morphology']}")
    for key, prod in (("p_init", config.CA_PARAMS["p_init"]),
                      ("gamma", config.CA_PARAMS["gamma"]),
                      ("theta", config.CA_PARAMS["theta"]),
                      ("decay_lambda", config.COOLING_PARAMS["decay_lambda"]),
                      ("cca_threshold", config.COOLING_PARAMS["cca_threshold"]),
                      ("competition_k", config.COOLING_PARAMS["competition_k"])):
        if s[key] != prod:
            departures.append(f"{key} {s[key]} (production {prod})")
    for key, prod in config.ACO_PARAMS.items():
        if key in s["aco_params"] and s["aco_params"][key] != prod:
            departures.append(f"{key} {s['aco_params'][key]} (production {prod})")

    if not departures:
        note("<strong>Production configuration.</strong> These settings are the "
             "ones behind the reported results.")
    else:
        note("<strong>Departed from the production configuration.</strong> "
             + "; ".join(departures) +
             ". Numbers below are exploratory and will not match the "
             "manuscript.")
    return departures


# ---------------------------------------------------------------- tab: grid
def tab_grid(ctx):
    comp = ctx.composition()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prohibited", f"{comp['n_p']}", f"{comp['pct_p']:.0f}% of domain")
    c2.metric("Available", f"{comp['n_a']}", f"{comp['pct_a']:.0f}% of domain")
    c3.metric("Vulnerable", f"{comp['n_v']}", f"{comp['pct_v']:.0f}% of domain")
    c4.metric("Plantable cells", f"{comp['n_plantable']}", "optimizer search space")

    note("The domain is a <strong>synthetic, non-georeferenced</strong> "
         "100 × 100 m block — 10 × 10 coarse planting cells over a 1 m "
         "evaluation grid. It is not a map of anywhere. There is no field site "
         "and no remote-sensing validation behind it.")

    left, right = st.columns(2)
    with left:
        chart(fg.land_use_map(ctx.grid), name="land_use_map")
    with right:
        chart(fg.equity_map(ctx.grid), name="equity_map")

    note("The equity map above uses a <strong>corrected index convention</strong>. "
         "The same map drawn by <code>legacy/AuditedCode_1.py</code> is its own "
         "transpose — mirrored about the diagonal. Only that one figure is "
         "affected; no score anywhere in this project reads that function.")


# ------------------------------------------------------------- tab: species
def tab_species(ctx):
    model = ctx.cooling_model
    left, right = st.columns([1.1, 1])
    with left:
        chart(fg.species_decay_curves(model), name="species_decay_curves")
    with right:
        chart(fg.species_potential(model), name="species_potential")

    rows = model.tree_species.summary_rows()
    st.markdown("#### Species parameters")
    st.dataframe(
        [{"Species": r["species"], "Binomial": r["binomial"],
          "Crown Ø (m)": round(r["crown_diameter_m"], 1),
          "Height (m)": round(r["height_m"], 1),
          "CPA (m²)": round(r["CPA_m2"], 1),
          "LAI adopted": round(r["LAI_adopted"], 2),
          "LAI from allometry": round(r["LAI_computed"], 4),
          "D_j": round(r["D_j"], 4)} for r in rows],
        width='stretch', hide_index=True)

    note("Two things this table makes visible and the manuscript does not. "
         "<strong>Adopted leaf-area values are not derived from the allometric "
         "constants published beside them</strong> — the allometric pipeline "
         "returns figures roughly two orders of magnitude smaller, and the "
         "analysis uses the adopted column. And <strong>three of the six "
         "species' heights are author estimates rather than measurements</strong>, "
         "sitting near the top of the observed field range.")


# ------------------------------------------------------------ tab: optimize
def tab_optimize(ctx, settings):
    st.markdown("#### Run the optimizer and watch it search")
    note("Ant Colony System. Each iteration, every ant builds a placement by "
         "choosing cells and species, and pheromone is deposited in proportion "
         "to the SECPI each placement earned. The chart fills in live — the gap "
         "between the colony mean and the best-so-far line is exploration "
         "against exploitation, as it happens.")

    p = settings["aco_params"]
    run = st.button(f"Run {p['n_iterations']} iterations × {p['n_ants']} ants",
                    type="primary")

    if run:
        placeholder = st.empty()
        progress = st.progress(0.0, text="Starting the colony…")
        n_iter = p["n_iterations"]
        last_paint = [0.0]

        def on_iteration(i, best, avg, hb, ha):
            # Called from inside the optimizer. Must not consume randomness.
            progress.progress((i + 1) / n_iter,
                              text=f"Iteration {i + 1} of {n_iter} — "
                                   f"best {max(hb):.4f}")
            # Throttle repainting so rendering does not dominate the run.
            now = time.time()
            if i == n_iter - 1 or now - last_paint[0] > 0.25:
                last_paint[0] = now
                placeholder.plotly_chart(
                    fg.convergence(list(hb), list(ha), n_iterations=n_iter),
                    width='stretch', config=PLOTLY_CONFIG,
                    key=f"live_{i}")

        from core.aco import run_once
        started = time.time()
        aco, result = run_once(
            ctx.grid, ctx.cooling_model, ctx.cutoffs,
            n_trees=p["n_trees"], seed=settings["aco_seed"],
            aco_params=p, on_iteration=on_iteration)
        elapsed = time.time() - started
        progress.empty()
        placeholder.empty()

        st.session_state["opt"] = {
            "result": result,
            "coords": aco.best_solution[0],
            "species": aco.best_solution[1],
            "cooling": np.asarray(aco.best_cooling, dtype=float).reshape(-1),
            "area_proportions": aco.best_area_proportions,
            "elapsed": elapsed,
            "settings_key": repr(settings),
        }

    state = st.session_state.get("opt")
    if not state:
        st.info("No run yet. Press the button above.")
        return

    if state.get("settings_key") != repr(settings):
        note("<strong>These results are from an earlier configuration.</strong> "
             "The sidebar has changed since this run. Re-run to match.")

    result = state["result"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SECPI (raw)", f"{result['raw_secpi']:.4f}")
    c2.metric("SECPI (0–5 scale)", f"{result['normalized_secpi']:.3f}")
    c3.metric("Species used", f"{result['species_actually_used']}",
              f"of {len(ctx.species_list)} available")
    c4.metric("Run time", f"{state['elapsed']:.1f} s",
              f"{result['n_trees_placed']} trees placed")

    note(f"The 0–5 scale is a min–max transform against derived theoretical "
         f"bounds, so it does <strong>not</strong> put zero at the "
         f"no-intervention baseline — a raw score of 0 maps to "
         f"{BASELINE_NORMALIZED:.3f}, not 0. The optimizer searches on the raw "
         f"value; the transform is strictly increasing, so it never changes "
         f"which placement wins.")

    chart(fg.convergence(result["history_best"], result["history_avg"],
                         title="Convergence",
                         n_iterations=p["n_iterations"],
                         final_secpi=result["raw_secpi"]), name="convergence")

    rule()
    left, right = st.columns(2)
    with left:
        chart(fg.placement_map(ctx.grid, state["coords"], state["species"],
                               ctx.cooling_model.tree_species), name="placement_map")
    with right:
        chart(fg.cooling_field(ctx.grid, state["cooling"], state["coords"],
                               state["species"], ctx.cooling_model.tree_species), name="cooling_field")

    rule()
    left, right = st.columns([1, 1.4])
    with left:
        chart(fg.cooling_class_distribution(state["area_proportions"]), name="cooling_class_distribution")
    with right:
        chart(fg.zonal_efficiency(state["cooling"],
                                  ctx.grid.vulnerability_weights), name="zonal_efficiency")

    with st.expander("The placement, tree by tree"):
        st.dataframe(
            [{"#": i + 1, "Species": sp,
              "X (m)": round(float(xy[0]), 1), "Y (m)": round(float(xy[1]), 1),
              "Crown radius (m)":
                  round(ctx.cooling_model.tree_species.get_crown_radius(sp), 1)}
             for i, (xy, sp) in enumerate(zip(state["coords"], state["species"]))],
            width='stretch', hide_index=True)


# ------------------------------------------------------------ tab: scenarios
def tab_scenarios(ctx, settings):
    st.markdown("#### Does equity weighting change where the trees go?")
    note("Two arms over the same grid. <strong>With</strong> keeps the "
         "vulnerable zones and their equity weights; <strong>without</strong> "
         "converts them to prohibited, flattening every weight to 1.0. The "
         "plantable set is identical in both, so the only thing that changes is "
         "the weighting. Both arms are scored against the same base-grid "
         "vulnerable mask — scoring the without arm against its own erased "
         "zones would return zero by construction.")

    c1, c2 = st.columns([1, 1])
    k_values = c1.multiselect("Tree counts to sweep", config.K_VALUES,
                              default=config.K_VALUES, key="w_ksweep")
    n_runs = c2.slider("Seeded restarts per configuration", 1, 10,
                       config.N_RUNS, 1, key="w_nruns")

    total = len(k_values) * 2 * n_runs
    est = total * 1.8
    st.caption(f"{total} optimizer runs — roughly "
               f"{est / 60:.1f} min at production settings.")

    if st.button(f"Run the sweep ({total} runs)", type="primary"):
        sctx = ScenarioContext(ctx.grid, ctx.cooling_model, ctx.cutoffs)
        bar = st.progress(0.0, text="Starting…")

        def progress(done, total_, label):
            bar.progress(done / total_, text=f"{label}  ({done}/{total_})")

        out = run_k_sweep(sctx, k_values=sorted(k_values), n_runs=n_runs,
                          aco_params=settings["aco_params"], progress=progress)
        bar.empty()
        st.session_state["sweep"] = out

    out = st.session_state.get("sweep")
    if not out:
        st.info("No sweep yet. Press the button above.")
        return

    summary = out["summary"]
    contrast = summary.get("arm_contrast_by_k", [])
    if contrast:
        mean_gain = float(np.mean([c["percent_of_with"] for c in contrast]))
        c1, c2, c3 = st.columns(3)
        best = max(contrast, key=lambda c: c["difference"])
        c1.metric("Mean equity gain", f"{mean_gain:.1f}%",
                  "across the swept tree counts")
        c2.metric("Largest gain", f"{best['percent_of_with']:.1f}%",
                  f"at k = {best['k']}")
        c3.metric("Restarts run", f"{len(out['restarts'])}")

    chart(fg.secpi_vs_k(summary), name="secpi_vs_k")
    rule()
    chart(fg.arm_comparison(summary), name="arm_comparison")
    rule()
    chart(fg.secpi_distribution(out["restarts"]), name="secpi_distribution")

    note("The box plot shows every individual restart, not just the means. "
         "Any difference between configurations has to clear that spread "
         "before it means anything.")

    with st.expander("Per-configuration numbers"):
        rows = []
        for arm in ("WITH", "WITHOUT"):
            for r in summary.get(arm, []):
                rows.append({
                    "Arm": arm, "k": r["k"], "n": r["n"],
                    "Mean": round(r["mean"], 4), "SD": round(r["sd"], 4),
                    "Min": round(r["min"], 4), "Max": round(r["max"], 4),
                    "Cooling share in vulnerable zones":
                        round(r["cooling_share_in_vulnerable_mean"], 4),
                    "Trees adjacent to vulnerable":
                        round(r["trees_adjacent_share_8conn_mean"], 3),
                })
        st.dataframe(rows, width='stretch', hide_index=True)


# ----------------------------------------------------------- tab: sensitivity
def tab_sensitivity(ctx, settings):
    st.markdown("#### Which parameters actually move the result?")
    note("One-at-a-time: every factor is swept to both ends of its range with "
         "all others held at baseline. <strong>Each evaluation is a full "
         "optimizer run</strong>, so this is a batch analysis, not a click. "
         "Forty factors across four families — cooling model, weighting, "
         "species morphology, species allometry.")

    c1, c2 = st.columns(2)
    n_samples = c1.slider("Repeats per bound", 1, 3, 1, 1, key="w_oatsamples",
                          help="Production is 3. Each repeat triples the cost.")
    quick = c2.checkbox("Reduced fidelity (faster, not comparable)",
                        value=False, key="w_oatquick")

    aco_params = dict(settings["aco_params"])
    if quick:
        aco_params.update({"n_ants": 8, "n_iterations": 12})

    n_evals = 1 * n_samples + 40 * 2 * n_samples
    per_run = 0.35 if quick else 1.8
    st.caption(f"{n_evals} optimizer runs — roughly "
               f"{n_evals * per_run / 60:.1f} min.")

    if quick:
        note("<strong>Reduced fidelity is on.</strong> The optimizer is running "
             "at 8 ants × 12 iterations instead of the production 20 × 40. "
             "Results are for exploring the shape of the ranking only and are "
             "<strong>not</strong> comparable with any reported number.")

    if st.button(f"Run the sweep ({n_evals} runs)", type="primary"):
        analyzer = SensitivityAnalyzer(ctx.grid, aco_params,
                                       reference_cutoffs=ctx.cutoffs)
        bar = st.progress(0.0, text="Establishing baseline…")

        def progress(done, total_, label):
            bar.progress(done / total_, text=f"{label}  ({done}/{total_})")

        np.random.seed(settings["grid_seed"])
        results = analyzer.run_oat_analysis(n_samples=n_samples, progress=progress)
        bar.empty()
        st.session_state["oat"] = {
            "results": results,
            "baseline": analyzer.baseline_secpi,
            "categories": analyzer.category_totals(),
            "quick": quick,
        }

    state = st.session_state.get("oat")
    if not state:
        st.info("No sweep yet. Press the button above.")
        return

    if state["quick"]:
        note("<strong>These results were produced at reduced fidelity.</strong>")

    ranked = sorted(state["results"], key=lambda r: r["sensitivity_index"],
                    reverse=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline SECPI", f"{state['baseline']:.4f}")
    c2.metric("Most sensitive", ranked[0]["parameter"],
              f"index {ranked[0]['sensitivity_index']:.4f}")
    c3.metric("Factors swept", f"{len(state['results'])}")

    chart(fg.sensitivity_tornado(state["results"]), name="sensitivity_tornado")
    rule()
    left, right = st.columns([1, 1.2])
    with left:
        chart(fg.sensitivity_by_category(state["categories"]), name="sensitivity_by_category")
    with right:
        chart(fg.sensitivity_ranges(state["results"], state["baseline"]), name="sensitivity_ranges")

    with st.expander("All forty factors"):
        st.dataframe(
            [{"Parameter": r["parameter"],
              "Category": r["category"].replace("_", " "),
              "SECPI low": round(r["secpi_low"], 4),
              "SECPI high": round(r["secpi_high"], 4),
              "Sensitivity index": round(r["sensitivity_index"], 5)}
             for r in ranked],
            width='stretch', hide_index=True)


# ---------------------------------------------------------------- tab: about
def tab_about():
    st.markdown("""
#### What this is

`lilim` is the consolidated, interactive reference implementation of SECPI —
the Synergistic and Equitable Cooling Performance Index — for optimising urban
tree placement across six Philippine tree functional types.

It is a rewrite of `legacy/AuditedCode_1.py`, reorganised into readable modules
and given the interface the original never had. Every number it produces is
verified equal, to the last bit, against that reference implementation.

#### What it is not

It is not the audited reference of record. `legacy/AuditedCode_1.py` remains
that, and where the two ever disagree, the disagreement is a defect here until
proven otherwise — with exactly one documented exception, described below.

It is also not a map of anywhere. The domain is synthetic and non-georeferenced.

#### The one deliberate difference

The coarse equity-weight map is computed with a corrected index convention.
The reference implementation indexes a fine-resolution array row-major when
that array is ordered x-major, so the map it draws is exactly its own
transpose — mirrored about the diagonal. `lilim` uses the correct index and
validates it against a ground truth derived from coordinates alone.

Correcting it in the reference implementation is a change to an audited
codebase and is still awaiting author sign-off there. Nothing else is affected:
no score, placement, cooling field, or sensitivity result reads that function.

#### Where the real record lives

This code is the *what*. The *why* is in the project's own documents, and the
next group should start there rather than reverse-engineering intent from
source:

- **`docs/DECISIONS.md`** — every semantic change made to the reference
  implementation, why it was made, who authorised it, and which commit applied
  it. The fixed study-wide normalization, the tree-count sweep range, the
  species-height assumptions, the state-isolation fix in the sensitivity sweep
  all trace to entries there.
- **`docs/PROJECT_LOG.md`** — the append-only evidentiary trail. Where a
  decision's reasoning is not fully captured in the decision itself, it is here.

#### Verifying this yourself

    python lilim/parity/check_parity.py --report parity_report.md

runs identical scenarios through both codebases and compares by exact equality.
It should be the first thing you run and the first thing you re-run after any
change.
""")

    st.markdown("#### Known open items")
    note("These are open questions in the science, not bugs in this code. "
         "They are stated here because a tool that hides them is worse than "
         "one that does not exist.")
    for item in [
        "The coarse equity-weight map here is corrected; the audited reference "
        "implementation's version is still its own transpose, pending author "
        "sign-off on the fix. The two will disagree on that one figure.",
        "Three of the six species' heights are author estimates, not "
        "measurements, and sit near the top of the observed field range.",
        "Per-species leaf-area values are adopted figures, not derived from the "
        "allometric constants published beside them. The allometric pipeline "
        "exists in the code but is not what the analysis uses.",
        "The sensitivity-analysis numbers in the originally published "
        "manuscript are not reproducible from this code. Current values come "
        "from re-running the analysis, not from reconciling with the original.",
        "The 0–5 SECPI scale does not place the no-intervention baseline at "
        f"zero — a raw score of 0 maps to {BASELINE_NORMALIZED:.3f}.",
        "Equity weighting runs as three discrete levels (1.0 / 1.5 / 2.0) and "
        "has not been reconciled against the manuscript table that describes a "
        "continuous 0.5–2.0 range.",
        "The domain is synthetic and non-georeferenced. There is no field site, "
        "no remote-sensing validation, and no raster of any real city.",
        "The optimizer is Ant Colony System with a pseudo-random-proportional "
        "rule. It has no diversity-enforcement mechanism, so a converged "
        "single-species solution reflects the objective, not a constraint.",
        "Multi-grid Morris sensitivity and morphological robustness validation "
        "are not in this app. They are batch analyses and live in `tools/`.",
    ]:
        st.markdown(f"- {item}")


# ------------------------------------------------------------------- main
def main():
    settings = sidebar()

    st.markdown("# lilim")
    st.markdown(
        "*Synergistic and Equitable Cooling Performance Index — an interactive "
        "reference implementation for optimising urban tree placement across "
        "six Philippine tree functional types.*")

    production_badge(settings)

    ctx = cached_context(
        settings["grid_seed"], settings["morphology"], settings["p_init"],
        settings["gamma"], settings["theta"], settings["decay_lambda"],
        settings["cca_threshold"], settings["competition_k"])

    q1, q2, q3 = ctx.cutoffs
    st.caption(
        f"Study-wide cooling cutoffs, calibrated once and reused everywhere:  "
        f"Q1 {q1:.3e}   Q2 {q2:.3e}   Q3 {q3:.3e}")

    tabs = st.tabs(["Grid", "Species & cooling", "Optimize",
                    "Scenarios", "Sensitivity", "About"])
    with tabs[0]:
        tab_grid(ctx)
    with tabs[1]:
        tab_species(ctx)
    with tabs[2]:
        tab_optimize(ctx, settings)
    with tabs[3]:
        tab_scenarios(ctx, settings)
    with tabs[4]:
        tab_sensitivity(ctx, settings)
    with tabs[5]:
        tab_about()


main()
