"""Every chart `lilim` can draw, one function per figure.

Adapted from the plotting logic scattered across `legacy/AuditedCode_1.py` --
`EnhancedVisualizer` (:2229-2630), `SensitivityAnalyzer.plot_sensitivity_results`
(:1107-1246), `SuboptimalScenariosGenerator._plot_k_comparison` /
`_plot_secpi_vs_k_curve` (:3034-3165), and the convergence plot written inline
in `main_revised_validation()` (:3506-3519). The reference implementation had no
single visualization module; that is why this one exists.

Two rules hold throughout:

1. **Nothing here computes science.** Every function takes already-computed
   arrays and renders them. If a figure needs a derived quantity, it is derived
   in `core/` and passed in.

2. **Nothing here consumes randomness.** These may be called from inside an
   optimizer callback, where a single stray draw would change the result.

Every figure returns a `plotly.graph_objects.Figure` and inherits the theme
registered in `theme.py`, so styling lives in exactly one place.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme

theme.register()


# ---------------------------------------------------------------- captions
# Captions live HERE, not inside the figures.
#
# They used to be Plotly annotations positioned in paper coordinates below the
# plot. That could not work: a paper offset scales with each figure's own
# height, while the template's bottom margin was fixed, so captions landed in a
# different place on every figure and most were clipped -- and on the cooling
# field the caption came down on top of the legend. Rendered as page text
# instead, they reflow at any width and cannot be cropped.
#
# The export path re-attaches the caption to the figure (see `export.py`), so a
# saved image still carries its caveat. There the bottom margin is computed
# from the wrapped line count rather than assumed.
CAPTIONS: dict[str, str] = {
    "convergence":
        "Best-so-far is monotone by construction; the colony mean is not. The gap "
        "between them is exploration against exploitation.",
    "land_use_map":
        "A synthetic, non-georeferenced 100 x 100 m domain: 10 x 10 coarse planting "
        "cells over a 1 m evaluation grid. It is not a map of anywhere.",
    "equity_map":
        "Three discrete levels: 2.0 within 10 m of a vulnerable cell, 1.5 within "
        "20 m, 1.0 beyond. Prohibited cells marked --. Uses the corrected index "
        "convention; the audited reference implementation's version of this map is "
        "its own transpose.",
    "species_decay_curves":
        "Gaussian decay exp(-lambda (d/C_D)^2), scaled by each species' own crown "
        "diameter, so wider crowns reach further. Single tree, no competition.",
    "species_potential":
        "Leaf-area values are the adopted per-species figures, not derived from the "
        "allometric constants published beside them.",
    "cooling_field":
        "Colour scale is clipped at the 99th percentile so a single hot crown centre "
        "does not flatten the rest of the field. Crowns are drawn at true metric "
        "radius. The grey discs are not missing data: the crown-competition term "
        "saturates to exactly zero under any canopy, so the model delivers no "
        "cooling directly beneath a tree. See the known-open-items in the README.",
    "placement_map":
        "Crowns are drawn at true metric radius over the land-use classes, so "
        "overlap between neighbouring trees is shown to scale.",
    "cooling_class_distribution":
        "Classes are fixed study-wide cutoffs, not per-scenario quartiles, so these "
        "shares are comparable between configurations.",
    "zonal_efficiency":
        "Left: where the cooling landed. Right: what that placement earned once "
        "equity weighting is applied.",
    "secpi_vs_k":
        "Error bars are the standard deviation across seeded restarts at that "
        "configuration -- the stochastic noise any difference must clear.",
    "arm_comparison":
        "The two arms share a grid and a plantable set; only the equity weighting "
        "differs. Both are scored against the same base-grid vulnerable mask.",
    "secpi_distribution":
        "Every individual restart is plotted, not just the means. A difference "
        "between configurations has to clear this spread before it means anything.",
    "palette_size_effect":
        "Palette size is how many species the optimizer may choose from. It is not "
        "how many it actually uses, which is usually fewer.",
    "sensitivity_tornado":
        "One-at-a-time: each factor swept to both bounds with every other held at "
        "baseline.",
    "sensitivity_by_category":
        "Totals are sums, so a category with more parameters in it accumulates more "
        "by construction. Compare with the ranking above.",
    "sensitivity_ranges":
        "Each bar spans the SECPI produced at that factor's low and high bounds, "
        "against the baseline marked by the dashed line.",
}


# ------------------------------------------------------------------ helpers
def _empty(message: str) -> go.Figure:
    """A placeholder that says why there is nothing to show."""
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=message, showarrow=False,
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          font=dict(family=theme.FONT_SANS, size=13,
                                    color=theme.INK_SOFT))],
        height=280,
    )
    return fig


def _square_axes(fig, width_m, height_m):
    """Lock a map figure to true 1:1 metric aspect, so circles are circles."""
    fig.update_xaxes(range=[0, width_m], constrain="domain", title_text="X (m)")
    fig.update_yaxes(range=[0, height_m], scaleanchor="x", scaleratio=1,
                     constrain="domain", title_text="Y (m)")


# ---------------------------------------------------------- ACO convergence
def convergence(history_best, history_avg, title="Optimizer convergence",
                n_iterations=None, final_secpi=None) -> go.Figure:
    """Best and mean SECPI per iteration -- the search, as it happened.

    This is the figure the manuscript-regeneration pipeline never drew, even
    though the optimizer has always returned both traces. Best-so-far is
    monotone by construction; the mean is not, and the gap between them is the
    live picture of exploration against exploitation.
    """
    if not history_best:
        return _empty("Run the optimizer to see it converge.")

    iterations = list(range(1, len(history_best) + 1))
    running_best = np.maximum.accumulate(np.asarray(history_best, dtype=float))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations, y=history_avg, name="Colony mean",
        mode="lines", line=dict(color=theme.SLATE, width=1.5, dash="dot"),
        hovertemplate="iteration %{x}<br>mean %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=history_best, name="Iteration best",
        mode="lines", line=dict(color=theme.CANOPY_LIGHT, width=1.5),
        hovertemplate="iteration %{x}<br>best %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=running_best, name="Best so far",
        mode="lines", line=dict(color=theme.CANOPY, width=2.5),
        hovertemplate="iteration %{x}<br>best so far %{y:.4f}<extra></extra>",
    ))

    # Keep the x-axis fixed to the full run so the chart does not rescale
    # under the viewer while it is still filling in.
    if n_iterations:
        fig.update_xaxes(range=[0.5, n_iterations + 0.5])

    if final_secpi is not None:
        fig.add_hline(y=final_secpi, line=dict(color=theme.RULE, width=1, dash="dash"))

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="SECPI (raw)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1.0),
        height=theme.H_SHORT,
    )
    return fig


# ----------------------------------------------------------------- the grid
def land_use_map(grid, title="Land use") -> go.Figure:
    """The coarse grid, one cell per planting unit, coloured by class."""
    from core import config

    coarse = grid.coarse_grid
    h, w = coarse.shape
    codes = [config.P_CODE, config.A_CODE, config.V_CODE]

    # Map the sparse class codes onto 0..n-1 so a discrete colorscale lines up.
    lookup = {code: i for i, code in enumerate(codes)}
    z = np.vectorize(lambda v: lookup.get(int(v), 0))(coarse).astype(float)

    n = len(codes)
    colorscale = []
    for i, code in enumerate(codes):
        colorscale.append([i / n, theme.LAND_USE_COLORS[code]])
        colorscale.append([(i + 1) / n, theme.LAND_USE_COLORS[code]])

    label = np.array([[theme.LAND_USE_LABELS[int(coarse[i, j])]
                       for j in range(w)] for i in range(h)])

    cell = grid.coarse_cell_size
    fig = go.Figure(go.Heatmap(
        z=z,
        x=[(j + 0.5) * cell for j in range(w)],
        y=[(i + 0.5) * cell for i in range(h)],
        colorscale=colorscale, zmin=-0.5, zmax=n - 0.5,
        showscale=False, xgap=1, ygap=1,
        customdata=label,
        hovertemplate="cell (%{x:.0f}, %{y:.0f}) m<br>%{customdata}<extra></extra>",
    ))

    # A real legend, since a heatmap gives none. Invisible scatter proxies.
    for code in codes:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=theme.LAND_USE_COLORS[code], symbol="square"),
            name=theme.LAND_USE_LABELS[code], showlegend=True,
        ))

    fig.update_layout(
        title=title, height=theme.H_MAP,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0),
    )
    _square_axes(fig, grid.fine_width, grid.fine_height)
    return fig


def equity_map(grid, title="Equity weights (coarse cells)") -> go.Figure:
    """Mean equity weight per coarse cell, with the value printed in each.

    Uses the corrected index convention -- see `core/grid.py`. The reference
    implementation's version of this map is its own transpose.
    """
    from core import config

    weights = grid.get_coarse_cell_weights()
    coarse = grid.coarse_grid
    h, w = coarse.shape
    cell = grid.coarse_cell_size

    text = np.empty((h, w), dtype=object)
    for i in range(h):
        for j in range(w):
            text[i, j] = ("--" if coarse[i, j] == config.P_CODE
                          else f"{weights[i, j]:.2f}")

    # Three discrete levels, so a stepped scale rather than a smooth ramp.
    colorscale = [
        [0.00, theme.EQUITY_COLORS[1.0]], [0.33, theme.EQUITY_COLORS[1.0]],
        [0.33, theme.EQUITY_COLORS[1.5]], [0.67, theme.EQUITY_COLORS[1.5]],
        [0.67, theme.EQUITY_COLORS[2.0]], [1.00, theme.EQUITY_COLORS[2.0]],
    ]

    fig = go.Figure(go.Heatmap(
        z=weights,
        x=[(j + 0.5) * cell for j in range(w)],
        y=[(i + 0.5) * cell for i in range(h)],
        colorscale=colorscale, zmin=1.0, zmax=2.0,
        xgap=1, ygap=1,
        text=text, texttemplate="%{text}",
        textfont=dict(family=theme.FONT_MONO, size=10, color=theme.INK),
        colorbar=dict(title=dict(text="weight", side="right"),
                      tickvals=[1.0, 1.5, 2.0], thickness=12, len=0.6),
        hovertemplate="cell (%{x:.0f}, %{y:.0f}) m<br>weight %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(title=title, height=theme.H_MAP)
    _square_axes(fig, grid.fine_width, grid.fine_height)
    return fig


# ------------------------------------------------------------- cooling model
def species_decay_curves(cooling_model, max_distance_m=50.0,
                         title="Cooling decay by species") -> go.Figure:
    """One radial decay profile per species, overlaid for comparison.

    The reference implementation drew these as six separate panels. Overlaid,
    the ordering by crown diameter is immediately legible -- which is the whole
    point of the figure.
    """
    ts = cooling_model.tree_species
    fig = go.Figure()

    for name in ts.species_list:
        distances, cooling = cooling_model.decay_curve(name, max_distance_m)
        radius = ts.get_crown_radius(name)
        fig.add_trace(go.Scatter(
            x=distances, y=cooling, name=name, mode="lines",
            line=dict(color=ts.get_species_color(name), width=2),
            hovertemplate=(f"<b>{name}</b><br>distance %{{x:.1f}} m"
                           f"<br>cooling %{{y:.4f}}"
                           f"<br>crown radius {radius:.1f} m<extra></extra>"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Distance from tree (m)",
        yaxis_title="Cooling contribution",
        height=theme.H_CHART,
    )
    return fig


def species_potential(cooling_model, title="Cooling potential by species") -> go.Figure:
    """The normalized potential D_j, split into its shade and evaporation parts."""
    ts = cooling_model.tree_species
    rows = ts.summary_rows()
    rows.sort(key=lambda r: r["D_j"])

    names = [r["species"] for r in rows]
    shade = [ts.shade_weight * r["CPA_m2"] / ts.max_CPA for r in rows]
    evap = [ts.evap_weight * r["LAI_adopted"] / ts.max_LAI for r in rows]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=shade, orientation="h", name=f"Crown area ({ts.shade_weight:g})",
        marker=dict(color=theme.CANOPY),
        hovertemplate="%{y}<br>shade term %{x:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=names, x=evap, orientation="h", name=f"Leaf area ({ts.evap_weight:g})",
        marker=dict(color=theme.CANOPY_LIGHT),
        hovertemplate="%{y}<br>evaporation term %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title, barmode="stack",
        xaxis_title="Normalized cooling potential  D_j",
        height=theme.H_SHORT,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    return fig


# ------------------------------------------------------------ the solution
def cooling_field(grid, cooling_values, tree_placements=None,
                  tree_species_list=None, tree_species_obj=None,
                  title="Delivered cooling") -> go.Figure:
    """The fine-grid cooling field, with crowns drawn at true metric radius.

    `cooling_values` is ordered x-major over `fine_grid_points`, so it reshapes
    to [ix, iy] and must be transposed to display as [row, col]. Getting this
    backwards mirrors the map -- the same index trap documented in `core/grid.py`.
    """
    cooling = np.asarray(cooling_values, dtype=float).reshape(-1)
    field = cooling.reshape(grid.n_cols_fine, grid.n_rows_fine).T

    vmax = float(np.percentile(cooling, 99)) or float(cooling.max()) or 1.0

    fig = go.Figure(go.Heatmap(
        z=field,
        x=grid.fine_x_coords, y=grid.fine_y_coords,
        colorscale=theme.COOLING_SCALE, zmin=0, zmax=vmax,
        colorbar=dict(title=dict(text="cooling", side="right"),
                      thickness=12, len=0.6),
        hovertemplate="(%{x:.0f}, %{y:.0f}) m<br>cooling %{z:.4f}<extra></extra>",
    ))
    # Cells the competition term has driven to zero, marked explicitly.
    #
    # These are REAL ZEROS, not missing data and not a drawing artefact. The
    # crown-competition damping is 1/(1+exp(K*(CCA - threshold))) with a
    # threshold of 1.2, but CCA accumulates crown area in SQUARE METRES and a
    # single crown contributes 70-450 m2. The logistic is fully saturated by
    # about 10 m2, so the factor is ~0.9975 outside any crown and exactly 0.0
    # inside one -- a step at the crown edge rather than a graded penalty. The
    # result is that a tree delivers no cooling at all directly beneath itself.
    #
    # Rendered against the near-white low end of the cooling ramp these zeros
    # were indistinguishable from "very little cooling", and read as white discs
    # obscuring the map. Giving them their own colour and legend entry is what
    # makes the model's behaviour legible instead of looking like a bug in the
    # plot. Fixing the model itself is a change to the reference implementation
    # and is not this module's call -- see the README's known-open-items.
    suppressed = (cooling <= 0.0)
    if suppressed.any():
        mask = np.where(suppressed, 1.0, np.nan)
        mask = mask.reshape(grid.n_cols_fine, grid.n_rows_fine).T
        fig.add_trace(go.Heatmap(
            z=mask,
            x=grid.fine_x_coords, y=grid.fine_y_coords,
            colorscale=[[0, theme.SUPPRESSED], [1, theme.SUPPRESSED]],
            showscale=False, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=theme.SUPPRESSED, symbol="square",
                        line=dict(color=theme.RULE, width=1)),
            name="no cooling delivered (crown competition)", showlegend=True,
        ))

    if tree_placements is not None and tree_species_obj is not None:
        shapes = []
        seen = set()
        for (tx, ty), name in zip(tree_placements, tree_species_list):
            r = tree_species_obj.get_crown_radius(name)
            color = tree_species_obj.get_species_color(name)
            # Circles in DATA coordinates, so a crown is drawn at its real
            # metric size and stays correct under zoom.
            shapes.append(dict(
                type="circle", xref="x", yref="y",
                x0=tx - r, x1=tx + r, y0=ty - r, y1=ty + r,
                line=dict(color=color, width=1.5),
                fillcolor="rgba(0,0,0,0)", layer="above",
            ))
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty], mode="markers",
                marker=dict(size=11, color=color,
                            line=dict(color=theme.PAPER, width=1.5)),
                name=name, legendgroup=name, showlegend=name not in seen,
                hovertemplate=(f"<b>{name}</b><br>(%{{x:.0f}}, %{{y:.0f}}) m"
                               f"<br>crown radius {r:.1f} m<extra></extra>"),
            ))
            seen.add(name)
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        title=title, height=theme.H_FIELD,
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0),
    )
    _square_axes(fig, grid.fine_width, grid.fine_height)
    return fig


def placement_map(grid, tree_placements, tree_species_list, tree_species_obj,
                  title="Tree placement") -> go.Figure:
    """Where the trees went, over the land-use classes, crowns at true radius."""
    fig = land_use_map(grid, title=title)

    shapes = []
    seen = set()
    for (tx, ty), name in zip(tree_placements, tree_species_list):
        r = tree_species_obj.get_crown_radius(name)
        color = tree_species_obj.get_species_color(name)
        shapes.append(dict(
            type="circle", xref="x", yref="y",
            x0=tx - r, x1=tx + r, y0=ty - r, y1=ty + r,
            line=dict(color=color, width=1.5),
            fillcolor=color, opacity=0.18, layer="above",
        ))
        fig.add_trace(go.Scatter(
            x=[tx], y=[ty], mode="markers",
            marker=dict(size=12, color=color,
                        line=dict(color=theme.INK, width=1.2)),
            name=name, legendgroup=name, showlegend=name not in seen,
            hovertemplate=(f"<b>{name}</b><br>(%{{x:.0f}}, %{{y:.0f}}) m"
                           f"<br>crown radius {r:.1f} m<extra></extra>"),
        ))
        seen.add(name)

    fig.update_layout(shapes=shapes)
    return fig


def cooling_class_distribution(area_proportions, mean_vuln_weights=None,
                               title="Cooling classes") -> go.Figure:
    """How the domain distributed across the four cooling classes.

    This is what SECPI actually scores -- the area in each class, weighted by
    class rank and by the mean equity weight of the cells that landed there.
    """
    if area_proportions is None:
        return _empty("Run the optimizer to see the class distribution.")

    labels = ["Class 1<br>lowest", "Class 2", "Class 3", "Class 4<br>highest"]
    colors = [theme.RULE, theme.CANOPY_LIGHT, theme.CANOPY, theme.CANOPY_DEEP]
    pct = [100.0 * p for p in area_proportions]

    hover = "%{x}<br>area %{y:.1f}%<extra></extra>"
    customdata = None
    if mean_vuln_weights is not None:
        customdata = list(mean_vuln_weights)
        hover = ("%{x}<br>area %{y:.1f}%"
                 "<br>mean equity weight %{customdata:.3f}<extra></extra>")

    fig = go.Figure(go.Bar(
        x=labels, y=pct, marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in pct], textposition="outside",
        textfont=dict(family=theme.FONT_MONO, size=11),
        customdata=customdata, hovertemplate=hover,
    ))
    fig.update_layout(
        title=title, yaxis_title="Share of domain (%)",
        height=theme.H_SHORT, showlegend=False,
    )
    fig.update_yaxes(range=[0, max(pct) * 1.2 if max(pct) else 1])
    return fig


def zonal_efficiency(cooling_values, vulnerability_weights,
                     title="Cooling by equity zone") -> go.Figure:
    """Mean delivered cooling per equity zone, and its weighted contribution.

    The left panel answers "where did the cooling land"; the right answers "how
    much did that placement earn once equity weighting is applied".
    """
    cooling = np.asarray(cooling_values, dtype=float).reshape(-1)
    weights = np.asarray(vulnerability_weights, dtype=float).reshape(-1)

    zones = [("Low priority (w=1.0)", 1.0, theme.EQUITY_COLORS[1.0]),
             ("Medium priority (w=1.5)", 1.5, theme.EQUITY_COLORS[1.5]),
             ("High priority (w=2.0)", 2.0, theme.EQUITY_COLORS[2.0])]

    names, means, sds, counts, weighted, colors = [], [], [], [], [], []
    for label, w, color in zones:
        if w == 2.0:
            mask = weights >= 2.0
        elif w == 1.5:
            mask = (weights >= 1.5) & (weights < 2.0)
        else:
            mask = weights < 1.5
        vals = cooling[mask]
        names.append(label)
        colors.append(color)
        means.append(float(vals.mean()) if vals.size else 0.0)
        sds.append(float(vals.std()) if vals.size else 0.0)
        counts.append(int(vals.size))
        weighted.append((float(vals.mean()) if vals.size else 0.0) * w)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("Mean cooling delivered",
                                        "Equity-weighted contribution"))

    fig.add_trace(go.Bar(
        x=names, y=means, error_y=dict(type="data", array=sds, color=theme.INK_SOFT,
                                       thickness=1, width=6),
        marker=dict(color=colors, line=dict(color=theme.RULE, width=1)),
        customdata=counts,
        hovertemplate="%{x}<br>mean %{y:.4f}<br>%{customdata} cells<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=names, y=weighted,
        marker=dict(color=colors, line=dict(color=theme.RULE, width=1)),
        hovertemplate="%{x}<br>weighted %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    for ann in fig.layout.annotations:
        ann.font = dict(family=theme.FONT_SANS, size=12, color=theme.INK_SOFT)

    fig.update_layout(title=title, height=theme.H_WIDE)
    fig.update_yaxes(title_text="Cooling intensity", row=1, col=1)
    fig.update_yaxes(title_text="Weighted cooling", row=1, col=2)
    return fig


# --------------------------------------------------------------- scenarios
def secpi_vs_k(summary, title="SECPI against tree count") -> go.Figure:
    """Mean SECPI at each k, both arms, with restart spread as error bars."""
    fig = go.Figure()

    for arm in ("WITH", "WITHOUT"):
        rows = summary.get(arm) or []
        if not rows:
            continue
        ks = [r["k"] for r in rows]
        means = [r["mean"] for r in rows]
        sds = [r["sd"] for r in rows]
        label = ("With equity weighting" if arm == "WITH"
                 else "Without equity weighting")
        fig.add_trace(go.Scatter(
            x=ks, y=means, name=label, mode="lines+markers",
            line=dict(color=theme.ARM_COLORS[arm], width=2.5),
            marker=dict(size=8),
            error_y=dict(type="data", array=sds, color=theme.ARM_COLORS[arm],
                         thickness=1, width=5),
            hovertemplate=(f"<b>{label}</b><br>k = %{{x}}"
                           "<br>mean %{y:.4f}<extra></extra>"),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Trees planted (k)",
        yaxis_title="SECPI (raw)",
        height=theme.H_CHART,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    fig.update_xaxes(dtick=1)
    return fig


def arm_comparison(summary, title="Equity weighting, arm by arm") -> go.Figure:
    """Grouped bars per k, plus the contrast as a share of the WITH arm."""
    rows_with = summary.get("WITH") or []
    rows_without = summary.get("WITHOUT") or []
    if not rows_with or not rows_without:
        return _empty("Run the scenario sweep to compare arms.")

    ks = [r["k"] for r in rows_with]
    contrast = {c["k"]: c for c in summary.get("arm_contrast_by_k", [])}

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                        column_widths=[0.62, 0.38],
                        subplot_titles=("Mean SECPI by arm",
                                        "Equity gain, % of the weighted arm"))

    for arm, rows in (("WITH", rows_with), ("WITHOUT", rows_without)):
        label = ("With equity weighting" if arm == "WITH"
                 else "Without equity weighting")
        fig.add_trace(go.Bar(
            x=ks, y=[r["mean"] for r in rows], name=label,
            marker=dict(color=theme.ARM_COLORS[arm]),
            error_y=dict(type="data", array=[r["sd"] for r in rows],
                         color=theme.INK_SOFT, thickness=1, width=5),
            hovertemplate=f"<b>{label}</b><br>k = %{{x}}<br>%{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

    pct = [contrast.get(k, {}).get("percent_of_with") for k in ks]
    fig.add_trace(go.Bar(
        x=ks, y=pct, showlegend=False,
        marker=dict(color=theme.TERRACOTTA),
        text=[f"{p:.1f}%" if p is not None else "" for p in pct],
        textposition="outside", textfont=dict(family=theme.FONT_MONO, size=10),
        hovertemplate="k = %{x}<br>%{y:.2f}% higher with weighting<extra></extra>",
    ), row=1, col=2)

    for ann in fig.layout.annotations:
        ann.font = dict(family=theme.FONT_SANS, size=12, color=theme.INK_SOFT)

    fig.update_layout(
        title=title, height=theme.H_WIDE, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="Trees planted (k)", dtick=1, row=1, col=1)
    fig.update_xaxes(title_text="Trees planted (k)", dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="SECPI (raw)", row=1, col=1)
    fig.update_yaxes(title_text="Difference (%)", row=1, col=2)
    return fig


def secpi_distribution(restarts, group_by="k",
                       title="SECPI across restarts") -> go.Figure:
    """Every individual restart score, so the spread is visible, not just a mean.

    Means hide how much of a difference is restart noise. This shows each run.
    """
    if not restarts:
        return _empty("Run the scenario sweep to see the distribution.")

    fig = go.Figure()
    for arm in ("WITH", "WITHOUT"):
        rows = [r for r in restarts if r.get("arm") == arm]
        if not rows:
            continue
        label = ("With equity weighting" if arm == "WITH"
                 else "Without equity weighting")
        fig.add_trace(go.Box(
            x=[r[group_by] for r in rows],
            y=[r["raw_secpi"] for r in rows],
            name=label, marker=dict(color=theme.ARM_COLORS[arm]),
            boxpoints="all", jitter=0.5, pointpos=0,
            marker_size=5, line=dict(width=1.2),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title, boxmode="group",
        xaxis_title="Trees planted (k)" if group_by == "k" else group_by,
        yaxis_title="SECPI (raw)",
        height=theme.H_WIDE,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    return fig


def palette_size_effect(subset_summary,
                        title="SECPI against palette size") -> go.Figure:
    """Mean SECPI by how many species were available to choose from."""
    fig = go.Figure()
    for arm in ("WITH", "WITHOUT"):
        block = subset_summary.get(arm)
        if not block:
            continue
        by_s = block["mean_by_palette_size"]
        label = ("With equity weighting" if arm == "WITH"
                 else "Without equity weighting")
        fig.add_trace(go.Scatter(
            x=list(by_s.keys()), y=list(by_s.values()),
            name=label, mode="lines+markers",
            line=dict(color=theme.ARM_COLORS[arm], width=2.5),
            marker=dict(size=8),
            hovertemplate=f"<b>{label}</b><br>palette size %{{x}}"
                          "<br>mean %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Species available (palette size s)",
        yaxis_title="Mean SECPI (raw)",
        height=theme.H_CHART,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
    )
    fig.update_xaxes(dtick=1)
    return fig


# ------------------------------------------------------------- sensitivity
def sensitivity_tornado(results, top=15,
                        title="Most sensitive parameters") -> go.Figure:
    """Sensitivity index per factor, ranked, coloured by category."""
    if not results:
        return _empty("Run the sensitivity sweep to see the ranking.")

    rows = sorted(results, key=lambda r: r["sensitivity_index"], reverse=True)[:top]
    rows.reverse()   # Plotly draws horizontal bars bottom-up.

    fig = go.Figure()
    seen = set()
    for row in rows:
        cat = row["category"]
        fig.add_trace(go.Bar(
            y=[row["parameter"]], x=[row["sensitivity_index"]], orientation="h",
            marker=dict(color=theme.CATEGORY_COLORS.get(cat, theme.INK_SOFT)),
            name=cat.replace("_", " "), legendgroup=cat,
            showlegend=cat not in seen,
            hovertemplate=(f"<b>%{{y}}</b><br>{cat.replace('_', ' ')}"
                           "<br>index %{x:.5f}<extra></extra>"),
        ))
        seen.add(cat)

    fig.update_layout(
        title=title,
        xaxis_title="Sensitivity index  |SECPI(high) - SECPI(low)| / baseline",
        height=max(theme.H_CHART, 26 * len(rows) + 150),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0),
    )
    fig.update_yaxes(automargin=True)
    return fig


def sensitivity_by_category(category_totals,
                            title="Sensitivity by category") -> go.Figure:
    """Summed sensitivity index per parameter family."""
    if not category_totals:
        return _empty("Run the sensitivity sweep to see category totals.")

    items = sorted(category_totals.items(), key=lambda kv: kv[1])
    labels = [k.replace("_", " ") for k, _ in items]
    values = [v for _, v in items]
    colors = [theme.CATEGORY_COLORS.get(k, theme.INK_SOFT) for k, _ in items]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker=dict(color=colors),
        text=[f"{v:.4f}" for v in values], textposition="outside",
        textfont=dict(family=theme.FONT_MONO, size=11),
        hovertemplate="%{y}<br>total %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, xaxis_title="Total sensitivity index",
        height=theme.H_SHORT, showlegend=False,
    )
    fig.update_xaxes(range=[0, max(values) * 1.18 if values else 1])
    fig.update_yaxes(automargin=True)
    return fig


def sensitivity_ranges(results, baseline_secpi, top=20,
                       title="SECPI range spanned by each parameter") -> go.Figure:
    """Dumbbell of the low and high SECPI each factor produced, against baseline."""
    if not results:
        return _empty("Run the sensitivity sweep to see parameter ranges.")

    rows = sorted(results, key=lambda r: r["sensitivity_index"], reverse=True)[:top]
    rows.reverse()

    fig = go.Figure()
    for row in rows:
        color = theme.CATEGORY_COLORS.get(row["category"], theme.INK_SOFT)
        fig.add_trace(go.Scatter(
            x=[row["secpi_low"], row["secpi_high"]],
            y=[row["parameter"], row["parameter"]],
            mode="lines+markers", showlegend=False,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            hovertemplate=(f"<b>{row['parameter']}</b><br>"
                           "SECPI %{x:.4f}<extra></extra>"),
        ))

    if baseline_secpi:
        fig.add_vline(x=baseline_secpi,
                      line=dict(color=theme.INK, width=1.2, dash="dash"),
                      annotation_text=f"baseline {baseline_secpi:.4f}",
                      annotation_position="top",
                      annotation_font=dict(family=theme.FONT_MONO, size=10,
                                           color=theme.INK_SOFT))

    fig.update_layout(
        title=title, xaxis_title="SECPI (raw)",
        height=max(theme.H_CHART, 24 * len(rows) + 150),
    )
    fig.update_yaxes(automargin=True)
    return fig
