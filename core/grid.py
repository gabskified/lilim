"""Two-level urban grid: cellular-automata land-use generation plus fine-grid
discretisation for cooling evaluation.

Ported from `legacy/AuditedCode_1.py:1248-1605`.

⚠️  DETERMINISM WARNING — READ BEFORE EDITING
---------------------------------------------
Reproducibility here rests entirely on the GLOBAL numpy random state and the
exact ORDER in which draws are consumed. `np.random.seed(s)` is set by the
caller; this module then consumes draws in a fixed sequence:

    shuffle(indices) -> per-cell random() in row-major order -> (top-up loop
    randint/randint/random) -> choice(seed centres)

Do NOT convert this to `np.random.default_rng`, do NOT vectorise the per-cell
Bernoulli trial, and do NOT reorder the loops. Any of those changes how many
draws are taken and in what order, which silently changes every downstream
number. The parity harness tests this by array equality against the reference
implementation, so a mistake here fails loudly -- but only if you run it.

INDEX CONVENTIONS — THE SUBTLE PART
-----------------------------------
Two different orderings coexist and are easy to confuse:

  * `coarse_grid[i, j]` and `fine_grid[row, col]` are  [y-index, x-index].
  * `fine_grid_points` is built x-major:  [(x, y) for x in xs for y in ys],
    so point index n = ix * n_rows_fine + iy.

`vulnerability_weights` is ordered over `fine_grid_points`, i.e. x-major. Any
code indexing into it from (row, col) must therefore compute
`col * n_rows_fine + row`, NOT `row * n_cols_fine + col`. On a square grid the
two differ by exactly a transpose, which is why the error is easy to miss.
See `get_coarse_cell_weights` below and the note in the README.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from . import config


class TwoLevelUrbanGrid:
    """Coarse planting grid (CA-generated) over a fine cooling-evaluation grid."""

    def __init__(self, coarse_width=None, coarse_height=None,
                 coarse_cell_size=None, fine_cell_size=None):
        self.coarse_width = coarse_width if coarse_width is not None else config.COARSE_GRID["width"]
        self.coarse_height = coarse_height if coarse_height is not None else config.COARSE_GRID["height"]
        self.coarse_cell_size = (coarse_cell_size if coarse_cell_size is not None
                                 else config.COARSE_GRID["cell_size"])

        self.fine_width = self.coarse_width * self.coarse_cell_size
        self.fine_height = self.coarse_height * self.coarse_cell_size
        self.fine_cell_size = fine_cell_size if fine_cell_size is not None else config.FINE_GRID["cell_size"]

        self.n_cols_fine = int(self.fine_width / self.fine_cell_size)
        self.n_rows_fine = int(self.fine_height / self.fine_cell_size)
        self.total_fine_cells = self.n_rows_fine * self.n_cols_fine

        self.fine_x_coords = np.arange(0, self.fine_width, self.fine_cell_size)
        self.fine_y_coords = np.arange(0, self.fine_height, self.fine_cell_size)
        # x-major -- see the index-conventions note in the module docstring.
        self.fine_grid_points = np.array([(x, y) for x in self.fine_x_coords
                                          for y in self.fine_y_coords])

        self.coarse_x_centers = np.arange(self.coarse_cell_size / 2,
                                          self.fine_width, self.coarse_cell_size)
        self.coarse_y_centers = np.arange(self.coarse_cell_size / 2,
                                          self.fine_height, self.coarse_cell_size)
        self.coarse_centers = np.array([(x, y) for x in self.coarse_x_centers
                                        for y in self.coarse_y_centers])

        self.coarse_grid = None
        self.fine_grid = None
        self.plantable_coords = []
        self.vulnerable_coords = []
        self.vulnerability_weights = None

    # ------------------------------------------------------------------ CA
    def get_moore_neighborhood(self, grid, x, y) -> int:
        """Count of Prohibited cells in the 8-neighbourhood of (x, y)."""
        rows, cols = grid.shape
        p_count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nx, ny = x + i, y + j
                if 0 <= nx < rows and 0 <= ny < cols:
                    if grid[nx, ny] == 1:
                        p_count += 1
        return p_count

    def generate_ca_archetype(self, params=None, morphology=None,
                              max_iterations=40, p_target_range=None,
                              v_target_range=None):
        """Three-phase land-use generation.

        PHASE 1 -- Almeida-style recursive multiplicative CA growth, giving
        realistic clustering structure:

            p_i^{kl}(t+1) = gamma * [ sum_{j in Omega_i} N_j^l(t) / 8 ] * p_i^{kl}(t)

        with k=0 (empty/available), l=1 (Prohibited/built), Omega_i the Moore
        neighbourhood, and gamma a global growth-rate constant. The probability
        is clipped to [0,1] and used as a Bernoulli trial.

        Note the right-hand side is p(t), not p(t+1). The manuscript as
        originally transcribed had p(t+1) on BOTH sides, which is degenerate --
        solving for it forces 0 whenever gamma*Omega != 1. The corrected
        first-order recursion is what runs here, and requires both p(t) > 0 and
        a nonzero neighbour fraction for growth, so isolated cells far from any
        built cluster do not spontaneously convert.

        A morphology-dependent weighting multiplies the neighbour fraction:
        organic boosts cells with >= theta built neighbours; linear boosts a
        central band. Both are standard zoning/suitability extensions.

        PHASE 2 -- deterministic density calibration. CA growth is stochastic
        and proceeds in multi-cell jumps, so it will not reliably land inside a
        narrow target band on its own. Cells are trimmed (lowest Moore
        connectivity first, preserving cluster cores) or added until:

            target_N_P = round(p_mid * N_total) + N_V

        The + N_V term compensates for Phase 3 converting that many Prohibited
        cells to Vulnerable.

        PHASE 3 -- target-based Vulnerable-zone carving. Zones grow cell by
        cell via 4-connected BFS from a small number of Prohibited-cell seeds,
        stopping exactly at N_V = round(v_mid * N_total). This is NOT the
        "30 m Chebyshev buffer" described in the manuscript's Methods -- that
        construction is geometrically incompatible with a 5-10% target at this
        grid size. The BFS lands inside the target band on every run,
        independent of grid size.

        This is a two-stage process -- stochastic growth for structure, then
        deterministic calibration to a target density -- not unconstrained
        emergent CA behaviour.
        """
        params = dict(config.CA_PARAMS) if params is None else params
        morphology = morphology or params.get("morphology", config.CA_PARAMS["morphology"])
        p_target_range = p_target_range or config.P_TARGET_RANGE
        v_target_range = v_target_range or config.V_TARGET_RANGE

        p_init = params.get("p_init", config.CA_PARAMS["p_init"])
        gamma = params.get("gamma", config.CA_PARAMS["gamma"])
        p0 = params.get("p0", config.CA_PARAMS["p0"])
        theta = params.get("theta", config.CA_PARAMS["theta"])

        total_cells = self.coarse_height * self.coarse_width
        p_low, p_high = p_target_range
        p_mid = (p_low + p_high) / 2
        v_low, v_high = v_target_range
        v_mid = (v_low + v_high) / 2
        n_v_target = int(round(v_mid * total_cells))

        # --- PHASE 1: Almeida-style recursive multiplicative growth ---
        self.coarse_grid = np.zeros((self.coarse_height, self.coarse_width), dtype=int)
        n_seeds = int(total_cells * p_init)
        indices = [(r, c) for r in range(self.coarse_height)
                   for c in range(self.coarse_width)]
        np.random.shuffle(indices)                      # RNG draw 1
        for r, c in indices[:n_seeds]:
            self.coarse_grid[r, c] = 1

        transition_prob = np.full((self.coarse_height, self.coarse_width), p0, dtype=float)

        for _ in range(max_iterations):
            density = np.sum(self.coarse_grid == 1) / total_cells
            if p_low <= density <= p_high:
                break
            new_grid = self.coarse_grid.copy()
            new_prob = transition_prob.copy()
            for r in range(self.coarse_height):
                for c in range(self.coarse_width):
                    if self.coarse_grid[r, c] == 0:
                        n_p = self.get_moore_neighborhood(self.coarse_grid, r, c)
                        omega = n_p / 8.0
                        if morphology == "organic":
                            if n_p >= theta:
                                omega *= 1.5
                        elif morphology == "linear":
                            if abs(r - self.coarse_height // 2) < 2:
                                omega *= 1.5
                        p_next = gamma * omega * transition_prob[r, c]
                        p_next = min(max(p_next, 0.0), 1.0)
                        new_prob[r, c] = p_next
                        if np.random.random() < p_next:   # RNG draw, per cell
                            new_grid[r, c] = 1
            self.coarse_grid = new_grid
            transition_prob = new_prob

        # --- PHASE 2: exact trim/top-up, compensating for Phase 3 ---
        target_n_p = int(round(p_mid * total_cells)) + n_v_target
        current_n_p = int(np.sum(self.coarse_grid == 1))

        if current_n_p > target_n_p:
            excess = current_n_p - target_n_p
            p_coords = np.argwhere(self.coarse_grid == 1)
            neighbor_counts = np.array([
                self.get_moore_neighborhood(self.coarse_grid, r, c) for r, c in p_coords
            ])
            order = np.argsort(neighbor_counts)   # revert lowest connectivity first
            for idx in order[:excess]:
                r, c = p_coords[idx]
                self.coarse_grid[r, c] = 0
        elif current_n_p < target_n_p:
            needed = target_n_p - current_n_p
            added = 0
            attempts = 0
            max_attempts = total_cells * 20
            while added < needed and attempts < max_attempts:
                attempts += 1
                r = np.random.randint(0, self.coarse_height)
                c = np.random.randint(0, self.coarse_width)
                if self.coarse_grid[r, c] == 0:
                    n_p = self.get_moore_neighborhood(self.coarse_grid, r, c)
                    if n_p > 0 or np.random.random() < 0.5:
                        self.coarse_grid[r, c] = 1
                        added += 1

        # --- PHASE 3: Vulnerable-zone carving, BFS from P-cell seeds ---
        p_cells = np.argwhere(self.coarse_grid == 1)
        v_count = 0
        if len(p_cells) > 0:
            n_seed_centers = max(1, int(total_cells * 0.02))
            seed_idx = np.random.choice(len(p_cells),
                                        min(n_seed_centers, len(p_cells)),
                                        replace=False)
            queue = [tuple(p_cells[i]) for i in seed_idx]
            visited = set()
            qi = 0
            while v_count < n_v_target and qi < len(queue):
                cr, cc = queue[qi]
                qi += 1
                if (cr, cc) in visited:
                    continue
                visited.add((cr, cc))
                if self.coarse_grid[cr, cc] != config.V_CODE:
                    self.coarse_grid[cr, cc] = config.V_CODE
                    v_count += 1
                if v_count >= n_v_target:
                    break
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < self.coarse_height and 0 <= nc < self.coarse_width
                            and (nr, nc) not in visited):
                        queue.append((nr, nc))

        self.coarse_grid[self.coarse_grid == 0] = config.A_CODE

        # Safety net only. Phases 1-3 guarantee target compliance; this fires
        # only for pathological parameter combinations (e.g. a p_target_range
        # spanning >95%).
        if config.A_CODE not in self.coarse_grid:
            empty_indices = np.random.choice(total_cells, int(total_cells * 0.2),
                                             replace=False)
            flat_grid = self.coarse_grid.flatten()
            flat_grid[empty_indices] = config.A_CODE
            self.coarse_grid = flat_grid.reshape(self.coarse_height, self.coarse_width)

        self._expand_to_fine_grid()

        self.plantable_coords = []
        for i in range(self.coarse_height):
            for j in range(self.coarse_width):
                if self.coarse_grid[i, j] == config.A_CODE:
                    X = self.coarse_x_centers[j]
                    Y = self.coarse_y_centers[i]
                    self.plantable_coords.append((X, Y))

        vulnerable_fine = np.argwhere(self.fine_grid == config.V_CODE)
        self.vulnerable_coords = np.array([
            (self.fine_x_coords[col], self.fine_y_coords[row])
            for row, col in vulnerable_fine
        ])

        self.plantable_coords = np.array(self.plantable_coords)
        self.vulnerability_weights = self._calculate_vulnerability_weights()

        return self.coarse_grid, self.fine_grid

    def _expand_to_fine_grid(self):
        """Replicate each coarse cell's class across its fine cells."""
        expansion_factor = int(self.coarse_cell_size / self.fine_cell_size)
        self.fine_grid = np.zeros((self.n_rows_fine, self.n_cols_fine), dtype=int)
        for i in range(self.coarse_height):
            for j in range(self.coarse_width):
                start_row = i * expansion_factor
                end_row = (i + 1) * expansion_factor
                start_col = j * expansion_factor
                end_col = (j + 1) * expansion_factor
                self.fine_grid[start_row:end_row, start_col:end_col] = self.coarse_grid[i, j]

    # ------------------------------------------------------------------ equity
    def _calculate_vulnerability_weights(self) -> np.ndarray:
        """Equity weight per fine grid point, by distance to the nearest V cell.

        Three discrete levels: 2.0 within 10 m, 1.5 within 20 m, 1.0 beyond.
        Coordinate-based, so it carries no index-convention hazard. This is the
        array the optimizer and every SECPI value actually depend on.
        """
        weights = np.ones(len(self.fine_grid_points))

        if len(self.vulnerable_coords) > 0:
            for i, point in enumerate(self.fine_grid_points):
                distances = cdist([point], self.vulnerable_coords, "euclidean")[0]
                min_dist = np.min(distances)
                if min_dist <= config.EQUITY_NEAR_RADIUS_M:
                    weights[i] = config.EQUITY_LEVELS[2]
                elif min_dist <= config.EQUITY_MID_RADIUS_M:
                    weights[i] = config.EQUITY_LEVELS[1]

        return weights

    def get_coarse_cell_weights(self) -> np.ndarray:
        """Mean equity weight per coarse cell, as a (height, width) array.

        ⚠️  THIS FUNCTION DIVERGES FROM `legacy/AuditedCode_1.py` BY DESIGN.

        The reference implementation computes `fine_idx = fr * n_cols_fine + fc`
        -- row-major -- while `vulnerability_weights` is ordered x-major over
        `fine_grid_points`. On the square study grid the two conventions differ
        by exactly a transpose, so the map it returns is its own mirror image
        about the diagonal (verified: max absolute difference 1.0, and
        `allclose(legacy, correct.T)` holds).

        The correct index is used here. Fixing it in the reference
        implementation is a semantic change that needs author authorization and
        is still pending there; this is a separate codebase, so it is fixed
        here directly. Nothing else diverges -- the defect's only consumer is
        the coarse equity-weight map, and no SECPI, optimizer, cooling, or
        sensitivity result reads this function.
        """
        coarse_weights = np.ones((self.coarse_height, self.coarse_width))
        expansion_factor = int(self.coarse_cell_size / self.fine_cell_size)

        for i in range(self.coarse_height):
            for j in range(self.coarse_width):
                start_row = i * expansion_factor
                end_row = (i + 1) * expansion_factor
                start_col = j * expansion_factor
                end_col = (j + 1) * expansion_factor

                cell_weights = []
                for fr in range(start_row, end_row):
                    for fc in range(start_col, end_col):
                        # x-major: column index strides by the number of ROWS.
                        fine_idx = fc * self.n_rows_fine + fr
                        if fine_idx < len(self.vulnerability_weights):
                            cell_weights.append(self.vulnerability_weights[fine_idx])

                if cell_weights:
                    coarse_weights[i, j] = np.mean(cell_weights)

        return coarse_weights

    def coarse_cell_weights_by_coordinate(self) -> np.ndarray:
        """Same quantity as `get_coarse_cell_weights`, derived from coordinates.

        Deliberately computed WITHOUT index arithmetic: it selects fine points
        by their (x, y) position inside each coarse cell. Used by the parity
        harness as an independent ground truth, so the corrected indexing above
        is validated against something that cannot share its failure mode.
        """
        out = np.ones((self.coarse_height, self.coarse_width))
        half = self.coarse_cell_size / 2
        pts = self.fine_grid_points
        for i in range(self.coarse_height):
            for j in range(self.coarse_width):
                cx = self.coarse_x_centers[j]
                cy = self.coarse_y_centers[i]
                sel = ((pts[:, 0] >= cx - half) & (pts[:, 0] < cx + half) &
                       (pts[:, 1] >= cy - half) & (pts[:, 1] < cy + half))
                if np.any(sel):
                    out[i, j] = float(np.mean(self.vulnerability_weights[sel]))
        return out

    # ------------------------------------------------------------------ arms
    def convert_vulnerable_to_prohibited(self):
        """The WITHOUT-equity arm: erase the Vulnerable class entirely.

        Every V cell becomes Prohibited, `vulnerable_coords` empties, and every
        equity weight drops to 1.0. Note this leaves the plantable set
        unchanged -- V cells were never plantable -- so the manipulation is
        purely to the weighting, not to the search space.
        """
        self.coarse_grid[self.coarse_grid == config.V_CODE] = config.P_CODE
        self._expand_to_fine_grid()
        self.vulnerable_coords = np.array([])
        self.vulnerability_weights = np.ones(len(self.fine_grid_points))

    def composition(self) -> dict:
        """Land-use counts and shares. Derived, for display and verification."""
        coarse = self.coarse_grid
        total = int(coarse.size)
        n_p = int((coarse == config.P_CODE).sum())
        n_a = int((coarse == config.A_CODE).sum())
        n_v = int((coarse == config.V_CODE).sum())
        return {
            "n_p": n_p, "n_a": n_a, "n_v": n_v,
            "total_cells": total,
            "n_plantable": int(len(self.plantable_coords)),
            "pct_p": 100.0 * n_p / total,
            "pct_a": 100.0 * n_a / total,
            "pct_v": 100.0 * n_v / total,
        }


def build_grid(grid_seed: int = config.DEFAULT_GRID_SEED,
               ca_params: dict | None = None,
               morphology: str | None = None,
               coarse_grid: dict | None = None,
               fine_grid: dict | None = None) -> TwoLevelUrbanGrid:
    """Seed the global RNG and generate one grid. The one place seeding happens.

    Seeding immediately before generation (rather than relying on ambient
    state) is what makes this safe to call repeatedly from an interactive app,
    where the script re-runs on every widget interaction.
    """
    ca_params = dict(config.CA_PARAMS) if ca_params is None else dict(ca_params)
    morphology = morphology or ca_params.get("morphology", config.CA_PARAMS["morphology"])
    coarse_grid = coarse_grid or config.COARSE_GRID
    fine_grid = fine_grid or config.FINE_GRID

    np.random.seed(grid_seed)
    grid = TwoLevelUrbanGrid(
        coarse_width=coarse_grid["width"],
        coarse_height=coarse_grid["height"],
        coarse_cell_size=coarse_grid["cell_size"],
        fine_cell_size=fine_grid["cell_size"],
    )
    grid.generate_ca_archetype(params=ca_params, morphology=morphology)
    return grid


def without_equity_arm(grid: TwoLevelUrbanGrid) -> TwoLevelUrbanGrid:
    """A copy of `grid` with the Vulnerable class converted to Prohibited.

    Built by copying arrays onto a fresh instance rather than by deep-copying,
    matching how the reference implementation constructs this arm. Consumes no
    randomness, so it can be called at any point without disturbing a seeded
    sequence.
    """
    other = TwoLevelUrbanGrid(
        coarse_width=grid.coarse_width,
        coarse_height=grid.coarse_height,
        coarse_cell_size=grid.coarse_cell_size,
        fine_cell_size=grid.fine_cell_size,
    )
    other.coarse_grid = grid.coarse_grid.copy()
    other.fine_grid = grid.fine_grid.copy()
    other.plantable_coords = grid.plantable_coords.copy()
    other.vulnerable_coords = (grid.vulnerable_coords.copy()
                               if len(grid.vulnerable_coords) > 0 else np.array([]))
    other.vulnerability_weights = grid.vulnerability_weights.copy()
    other.convert_vulnerable_to_prohibited()
    return other
