"""SECPI — the Synergistic and Equitable Cooling Performance Index.

Ported from `legacy/AuditedCode_1.py:26-75` (cutoff calibration) and
`:2034-2138` (evaluation, bounds, normalization).

HOW THE INDEX WORKS
-------------------
Every fine grid point is classified into one of four cooling classes by where
its delivered cooling falls against three cutoffs (Q1, Q2, Q3). SECPI then
scores the resulting class distribution against a no-intervention baseline,
weighting each class by its rank and by the mean equity weight of the cells
that landed in it:

    SECPI = sum_k ( A_k - baseline_k ) * W_k * We_k

with A_k the area proportion in class k, baseline = [1, 0, 0, 0] (everything in
the worst class), W = [1, 2, 3, 4], and We_k the mean equity weight of class k.

FIXED CUTOFFS, NOT SELF-NORMALIZING  (Option B)
-----------------------------------------------
The cutoffs are calibrated ONCE per study by Monte Carlo pooling of random
valid placements, then reused identically by every optimizer instance in the
run. This matters more than it looks. Under the earlier self-normalizing
scheme each scenario took its own quartiles, which drove every placement to
approximately 25/25/25/25 regardless of how much cooling it actually
delivered -- magnitude differences between scenarios were normalized away, and
comparing scenarios was meaningless. Fixed study-wide cutoffs are what make
SECPI comparable across the k sweep, the species-subset sweep, and the
sensitivity analysis. See docs/DECISIONS.md.

Passing `reference_cutoffs=None` falls back to the old per-scenario behaviour.
It is retained only so the difference can be demonstrated; do not report
numbers produced that way.
"""
from __future__ import annotations

import numpy as np

from . import config

# Closed-form theoretical bounds of the raw index.
#
# The objective is linear in A over the 4-simplex for fixed We, and linear in
# each We over its box [0.5, 2.0] for fixed A. The extrema of such a bilinear
# objective sit at a simplex vertex combined with box boundary values;
# enumerating all 4 vertices x 4 boundary combinations gives:
#
#     max = 7.5   at A_4 = 1 (every cell in class 4), We_1 = 0.5, We_4 = 2.0
#     min = -1.0  at A_2 = 1 (every cell in class 2), We_1 = 2.0, We_2 = 0.5
#
# The vertex A_1 = 1 -- the true no-intervention baseline -- yields exactly 0
# regardless of We, since A equals the baseline there.
SECPI_THEORETICAL_MIN = -1.0
SECPI_THEORETICAL_MAX = 7.5

CLASS_WEIGHTS = np.array([1, 2, 3, 4])
BASELINE_PROPORTIONS = np.array([1.0, 0.0, 0.0, 0.0])


def calibrate_reference_cutoffs(grid, cooling_model, species_list,
                                reference_n_trees=None,
                                n_trees_range=None,
                                n_samples=None,
                                random_seed=None):
    """Study-wide Q1/Q2/Q3 cooling cutoffs, by Monte Carlo pooling.

    Each of `n_samples` draws independently samples its own tree count
    uniformly from `n_trees_range` (inclusive), matching the k = 1..6 range
    used elsewhere, rather than anchoring the reference frame to a single k.
    Pass `reference_n_trees` instead to pin it to one count.

    ⚠️  Consumes global RNG state in a fixed order: seed -> randint(sizes) ->
    per-draw choice(cells) + choice(species). Do not reorder or vectorise.
    """
    n_trees_range = n_trees_range or config.CUTOFF_CALIB["n_trees_range"]
    n_samples = n_samples if n_samples is not None else config.CUTOFF_CALIB["n_samples"]
    random_seed = (random_seed if random_seed is not None
                   else config.CUTOFF_CALIB["random_seed"])

    if random_seed is not None:
        np.random.seed(random_seed)

    n_plantable = len(grid.plantable_coords)

    if reference_n_trees is not None:
        tree_count_choices = [min(reference_n_trees, n_plantable)] * n_samples
    else:
        lo, hi = n_trees_range
        tree_count_choices = np.random.randint(lo, hi + 1, size=n_samples)
        tree_count_choices = np.minimum(tree_count_choices, n_plantable)

    pooled_values = []
    for n_place in tree_count_choices:
        n_place = int(n_place)
        if n_place <= 0:
            continue
        idxs = np.random.choice(n_plantable, size=n_place, replace=False)
        coords = [grid.plantable_coords[i] for i in idxs]
        species = list(np.random.choice(species_list, size=n_place, replace=True))
        total_cooling, _ = cooling_model.calculate_total_cooling(
            coords, species, grid.fine_grid_points
        )
        pooled_values.append(total_cooling.flatten())

    pooled = np.concatenate(pooled_values)
    q1, q2, q3 = np.percentile(pooled, [25, 50, 75])
    return (q1, q2, q3)


def classify_cooling(cooling_flat, cutoffs):
    """Assign each fine cell a cooling class 1-4 against fixed cutoffs.

    Boundary convention: lower-inclusive, upper-exclusive on the top edge. This
    resolves ties -- including the fully degenerate all-zero case, i.e. a true
    no-tree baseline -- to class 1 (worst), which is what the
    baseline = [1, 0, 0, 0] assumption requires. The opposite convention would
    push every tied cell into class 4 (best), inverting the intended semantics.
    """
    q1, q2, q3 = cutoffs
    cooling_class = np.zeros_like(cooling_flat, dtype=int)
    cooling_class[cooling_flat <= q1] = 1
    cooling_class[(cooling_flat > q1) & (cooling_flat <= q2)] = 2
    cooling_class[(cooling_flat > q2) & (cooling_flat <= q3)] = 3
    cooling_class[cooling_flat > q3] = 4
    return cooling_class


def evaluate(cooling_flat, vulnerability_weights, cutoffs=None):
    """Raw SECPI plus its components, from an already-computed cooling field.

    Returns `(secpi, area_proportions, cooling_class, mean_vuln_weights)`.
    `cutoffs=None` falls back to this scenario's own quartiles -- see the
    module docstring for why that is not comparable across scenarios.
    """
    if cutoffs is not None:
        q1, q2, q3 = cutoffs
    else:
        q1, q2, q3 = np.percentile(cooling_flat, [25, 50, 75])

    cooling_class = classify_cooling(cooling_flat, (q1, q2, q3))

    total_cells = len(cooling_class)
    area_proportions = np.zeros(4)
    for k in range(4):
        area_proportions[k] = np.sum(cooling_class == (k + 1)) / total_cells

    mean_vuln_weights = np.zeros(4)
    for k in range(4):
        class_mask = (cooling_class == (k + 1))
        if np.any(class_mask):
            mean_vuln_weights[k] = np.mean(vulnerability_weights[class_mask])
        else:
            mean_vuln_weights[k] = 1.0

    secpi = np.sum(
        (area_proportions - BASELINE_PROPORTIONS) * CLASS_WEIGHTS * mean_vuln_weights
    )

    return secpi, area_proportions, cooling_class, mean_vuln_weights


def normalize(raw_secpi):
    """Map raw SECPI onto a bounded 0-5 scale by min-max against the bounds:

        SECPI_norm = 5 * (raw + 1.0) / 8.5

    This is a strictly increasing affine transform, so it never changes which
    configuration is optimal -- the optimizer searches on the raw value and
    this is for reporting only.

    ⚠️  It does NOT anchor the no-intervention baseline at zero. Raw 0, the
    baseline case, maps to 5*(0+1)/8.5 = 0.588, because the global theoretical
    minimum (-1.0) sits below the baseline (0) rather than at it. A scheme that
    keeps 0 at the baseline is possible but is a different, non-min-max
    transform. Reported values carry this caveat.
    """
    return 5.0 * (raw_secpi - SECPI_THEORETICAL_MIN) / (
        SECPI_THEORETICAL_MAX - SECPI_THEORETICAL_MIN
    )


BASELINE_NORMALIZED = normalize(0.0)   # ~= 0.588, not 0. See normalize().


def cooling_share_in_vulnerable(cooling_flat, v_fine_mask) -> float:
    """Proportion of total delivered cooling landing inside Vulnerable cells.

    `v_fine_mask` must be built over `fine_grid_points`, i.e. x-major -- use
    `vulnerable_fine_mask()` below rather than flattening `fine_grid` directly.
    """
    total = float(cooling_flat.sum())
    if total <= 0.0:
        return float("nan")
    return float(cooling_flat[v_fine_mask].sum() / total)


def vulnerable_fine_mask(grid) -> np.ndarray:
    """Boolean mask over `fine_grid_points` selecting Vulnerable cells.

    `fine_grid[row, col]` is [y, x] but `fine_grid_points` is x-major, so the
    mask is the TRANSPOSE flattened, not the array flattened. Getting this
    backwards is the same index-convention trap described in `grid.py`.
    """
    return grid.fine_grid.T.reshape(-1) == config.V_CODE


def adjacency_masks(grid) -> dict:
    """Coarse cells adjacent to a Vulnerable cell, 8- and 4-connected."""
    coarse = grid.coarse_grid
    h, w = coarse.shape
    is_v = coarse == config.V_CODE
    adj8 = np.zeros((h, w), dtype=bool)
    adj4 = np.zeros((h, w), dtype=bool)
    for i in range(h):
        for j in range(w):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and is_v[ni, nj]:
                        adj8[i, j] = True
                        if di == 0 or dj == 0:
                            adj4[i, j] = True
    return {"adj8": adj8, "adj4": adj4}


def trees_adjacent_share(coords, adj, cell_size) -> float:
    """Proportion of placed trees whose coarse cell abuts a Vulnerable cell."""
    if len(coords) == 0:
        return float("nan")
    hits = 0
    for xy in coords:
        x, y = float(xy[0]), float(xy[1])
        i, j = int(y // cell_size), int(x // cell_size)
        if adj[i, j]:
            hits += 1
    return hits / len(coords)
