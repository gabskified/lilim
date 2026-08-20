"""Ant Colony System optimizer for tree placement and species selection.

Ported from `legacy/AuditedCode_1.py:1836-2206`.

WHICH ACO THIS IS
-----------------
Ant Colony System (Dorigo & Gambardella), using the pseudo-random-proportional
action rule: with probability q0 the ant greedily exploits the best
(pheromone^alpha * heuristic^beta) pair available; otherwise it samples one by
roulette wheel. There is NO diversity-enforcement mechanism -- if the search
converges on a single species, that is a property of the objective and the
heuristic, not something the optimizer prevents.

Each ant builds a solution by choosing `n_trees` (cell, species) pairs without
replacement of cells. The heuristic is a cell's mean equity weight times the
species' normalized cooling potential, precomputed once.

⚠️  DETERMINISM — the RNG draw order in this class is:
      seed(random_seed) -> uniform(pheromone jitter) -> then, per ant per tree,
      random() for the q0 branch, and choice() only on the exploration branch.
    The greedy branch consumes exactly ONE draw; the exploration branch
    consumes two. Changing the branch structure changes the stream. See the
    warning in `grid.py`.
"""
from __future__ import annotations

import numpy as np

from . import config, secpi as secpi_mod


class AntColonySystemACO:
    """ACS over (plantable cell, species) pairs, maximising raw SECPI."""

    SECPI_THEORETICAL_MIN = secpi_mod.SECPI_THEORETICAL_MIN
    SECPI_THEORETICAL_MAX = secpi_mod.SECPI_THEORETICAL_MAX

    def __init__(self, two_level_grid, cooling_model, n_trees=None,
                 n_ants=None, n_iterations=None, evaporation_rate=None,
                 alpha=None, beta=None, q0=None, random_seed=None,
                 species_subset=None, reference_cutoffs=None):

        # Seeded FIRST, before the pheromone jitter below consumes any draw.
        if random_seed is not None:
            np.random.seed(random_seed)

        p = config.ACO_PARAMS
        self.grid = two_level_grid
        self.cooling_model = cooling_model
        self.n_trees = min(n_trees if n_trees is not None else p["n_trees"],
                           len(two_level_grid.plantable_coords))
        self.n_ants = n_ants if n_ants is not None else p["n_ants"]
        self.n_iterations = n_iterations if n_iterations is not None else p["n_iterations"]
        self.evaporation_rate = (evaporation_rate if evaporation_rate is not None
                                 else p["evaporation_rate"])
        self.alpha = alpha if alpha is not None else p["alpha"]
        self.beta = beta if beta is not None else p["beta"]
        self.q0 = q0 if q0 is not None else p["q0"]
        self.random_seed = random_seed

        self.plantable_coords = two_level_grid.plantable_coords
        self.n_plantable = len(self.plantable_coords)

        if self.n_plantable == 0:
            raise ValueError("No plantable cells in this grid.")

        self.vulnerability_weights = two_level_grid.vulnerability_weights.copy()
        self.tree_species = cooling_model.tree_species

        if species_subset is not None:
            self.species_list = list(species_subset)
        else:
            self.species_list = self.tree_species.species_list

        self.n_species = len(self.species_list)

        base_pheromone = 1.0 / (self.n_plantable * self.n_species)
        self.pheromone = np.ones((self.n_plantable, self.n_species)) * base_pheromone
        self.pheromone *= np.random.uniform(0.8, 1.2, self.pheromone.shape)

        self.best_solution = None
        self.best_secpi = -np.inf
        self.best_cooling = None
        self.best_cca = None
        self.best_area_proportions = None

        # Fixed study-wide cutoffs. None falls back to per-scenario quartiles,
        # which is not comparable across scenarios -- see secpi.py.
        self.reference_cutoffs = reference_cutoffs

        self.heuristic_cache = self._precompute_heuristics()

    # ------------------------------------------------------------- heuristic
    def _precompute_heuristics(self):
        """eta[cell, species] = species cooling potential x cell equity weight.

        The cell's equity weight is the mean over the fine points falling
        inside it, selected by coordinate rather than by index -- so this
        carries no index-convention hazard.
        """
        n_cells = len(self.plantable_coords)
        n_species = len(self.species_list)
        heuristic_matrix = np.zeros((n_cells, n_species))

        half_cell = self.grid.coarse_cell_size / 2
        fine_points = self.grid.fine_grid_points

        for cell_idx in range(n_cells):
            cell_x, cell_y = self.plantable_coords[cell_idx]

            in_cell = ((fine_points[:, 0] >= cell_x - half_cell) &
                       (fine_points[:, 0] < cell_x + half_cell) &
                       (fine_points[:, 1] >= cell_y - half_cell) &
                       (fine_points[:, 1] < cell_y + half_cell))

            vuln_weight = np.mean(self.vulnerability_weights[in_cell]) if \
                np.any(in_cell) else 1.0

            for species_idx in range(n_species):
                cooling_pot = self.tree_species.get_normalized_cooling_potential(
                    self.species_list[species_idx])
                heuristic_matrix[cell_idx, species_idx] = cooling_pot * vuln_weight

        return heuristic_matrix

    # ------------------------------------------------------------ one ant
    def construct_solution(self):
        """One ant's tour: n_trees (cell, species) pairs, cells without repeats."""
        selected_cells = []
        selected_species = []
        available_cells = list(range(self.n_plantable))

        for _tree_idx in range(self.n_trees):
            if not available_cells:
                break

            if np.random.random() < self.q0:
                # Exploitation: take the best available pair outright.
                best_value = -1
                best_cell_idx = None
                best_species_idx = None

                for cell_idx in available_cells:
                    for species_idx in range(self.n_species):
                        tau = self.pheromone[cell_idx, species_idx]
                        eta = self.heuristic_cache[cell_idx, species_idx]
                        value = (tau ** self.alpha) * (eta ** self.beta)
                        if value > best_value:
                            best_value = value
                            best_cell_idx = cell_idx
                            best_species_idx = species_idx

                selected_cell_idx = best_cell_idx
                selected_species_idx = best_species_idx
            else:
                # Exploration: roulette wheel over all available pairs.
                probabilities = []
                choices = []

                for cell_idx in available_cells:
                    for species_idx in range(self.n_species):
                        tau = self.pheromone[cell_idx, species_idx]
                        eta = self.heuristic_cache[cell_idx, species_idx]
                        prob = (tau ** self.alpha) * (eta ** self.beta)
                        probabilities.append(prob)
                        choices.append((cell_idx, species_idx))

                probabilities = np.array(probabilities)
                if probabilities.sum() > 0:
                    probabilities /= probabilities.sum()
                else:
                    probabilities = np.ones(len(probabilities)) / len(probabilities)

                choice_idx = np.random.choice(len(choices), p=probabilities)
                selected_cell_idx, selected_species_idx = choices[choice_idx]

            selected_cells.append(selected_cell_idx)
            selected_species.append(self.species_list[selected_species_idx])
            available_cells.remove(selected_cell_idx)

        solution_coords = [self.plantable_coords[idx] for idx in selected_cells]
        return solution_coords, selected_species

    # ------------------------------------------------------------ objective
    def evaluate_secpi(self, tree_placements, tree_species_list):
        """Raw SECPI for one placement, plus the fields behind it."""
        total_cooling, cca_values = self.cooling_model.calculate_total_cooling(
            tree_placements, tree_species_list, self.grid.fine_grid_points
        )
        cooling_flat = total_cooling.flatten()
        score, area_proportions, _cls, _we = secpi_mod.evaluate(
            cooling_flat, self.vulnerability_weights, self.reference_cutoffs
        )
        return score, total_cooling, cca_values, area_proportions

    @staticmethod
    def normalize_secpi(raw_secpi):
        """Raw SECPI on the reported 0-5 scale. See `secpi.normalize`."""
        return secpi_mod.normalize(raw_secpi)

    # ------------------------------------------------------------ pheromone
    def update_pheromones(self, solutions, secpi_scores):
        """Evaporate globally, then deposit in proportion to relative quality."""
        self.pheromone *= (1 - self.evaporation_rate)

        best_idx = np.argmax(secpi_scores)
        best_secpi_iter = secpi_scores[best_idx]

        for solution_idx, (tree_coords, tree_species) in enumerate(solutions):
            secpi_score = secpi_scores[solution_idx]
            if best_secpi_iter > 0:
                quality = secpi_score / best_secpi_iter
            else:
                quality = 0.1

            for coord, species_name in zip(tree_coords, tree_species):
                cell_idx = np.where(
                    (self.plantable_coords == coord).all(axis=1))[0][0]
                species_idx = self.species_list.index(species_name)
                self.pheromone[cell_idx, species_idx] += quality

    # ------------------------------------------------------------------ run
    def run(self, on_iteration=None):
        """Run the colony. Returns `(history_best, history_avg)` per iteration.

        `on_iteration(iteration, best, avg, history_best, history_avg)` is
        called after each iteration's pheromone update, if supplied. It exists
        so a UI can watch the search converge instead of only seeing a final
        number.

        ⚠️  The callback MUST NOT consume randomness -- doing so would shift
        every subsequent draw and change the result. With `on_iteration=None`
        this method is behaviourally identical to the reference
        implementation's `run()`; the parity harness verifies both paths give
        the same answer.
        """
        history_best = []
        history_avg = []

        for iteration in range(self.n_iterations):
            solutions_iter = []
            secpi_scores_iter = []

            for _ant in range(self.n_ants):
                tree_coords, tree_species = self.construct_solution()
                score, cooling, cca, area_props = self.evaluate_secpi(
                    tree_coords, tree_species)

                solutions_iter.append((tree_coords, tree_species))
                secpi_scores_iter.append(score)

                if score > self.best_secpi:
                    self.best_secpi = score
                    self.best_solution = (tree_coords, tree_species)
                    self.best_cooling = cooling
                    self.best_cca = cca
                    self.best_area_proportions = area_props

            self.update_pheromones(solutions_iter, secpi_scores_iter)

            history_best.append(np.max(secpi_scores_iter))
            history_avg.append(np.mean(secpi_scores_iter))

            if on_iteration is not None:
                on_iteration(iteration, history_best[-1], history_avg[-1],
                             history_best, history_avg)

        self.history_best = history_best
        self.history_avg = history_avg
        return history_best, history_avg

    # ------------------------------------------------------------- reporting
    def result(self) -> dict:
        """Everything worth keeping from a finished run."""
        if self.best_solution is None:
            return {}
        coords, species = self.best_solution
        cooling = np.asarray(self.best_cooling, dtype=float).reshape(-1)
        return {
            "raw_secpi": float(self.best_secpi),
            "normalized_secpi": float(secpi_mod.normalize(self.best_secpi)),
            "seed": self.random_seed,
            "n_trees_requested": int(self.n_trees),
            "n_trees_placed": int(len(coords)),
            "coords": [[float(x), float(y)] for x, y in coords],
            "species": list(species),
            "species_actually_used": int(len(set(species))),
            "area_proportions": (None if self.best_area_proportions is None
                                 else [float(v) for v in self.best_area_proportions]),
            "cooling_sum": float(cooling.sum()),
            "cooling_mean": float(cooling.mean()),
            "cooling_max": float(cooling.max()),
            "cooling_std": float(cooling.std()),
            "history_best": [float(v) for v in getattr(self, "history_best", [])],
            "history_avg": [float(v) for v in getattr(self, "history_avg", [])],
        }


def run_once(grid, cooling_model, cutoffs, n_trees=None, species_subset=None,
             seed=None, aco_params=None, on_iteration=None):
    """One seeded optimizer restart, returning the optimizer and its result.

    The single entry point the app and the sweeps both use, so seeding and
    parameter handling live in exactly one place.
    """
    p = dict(config.ACO_PARAMS)
    if aco_params:
        p.update(aco_params)

    aco = AntColonySystemACO(
        grid, cooling_model,
        n_trees=n_trees if n_trees is not None else p["n_trees"],
        n_ants=p["n_ants"],
        n_iterations=p["n_iterations"],
        evaporation_rate=p["evaporation_rate"],
        alpha=p["alpha"],
        beta=p["beta"],
        q0=p["q0"],
        random_seed=seed,
        species_subset=species_subset,
        reference_cutoffs=cutoffs,
    )
    aco.run(on_iteration=on_iteration)
    return aco, aco.result()
