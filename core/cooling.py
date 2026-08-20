"""Gaussian radial cooling model with crown-competition damping.

Ported from `legacy/AuditedCode_1.py:1755-1828`.

THE MODEL, IN THREE PARTS
-------------------------
1. Each tree's cooling falls off with a Gaussian radial decay scaled by its own
   crown diameter:

       cooling_j(d) = D_j * exp( -lambda * (d / C_D,j)^2 )

   Note the SQUARED term. The manuscript's Methods equation omits it while
   still calling the decay Gaussian; the word is right and the equation was
   wrong. `lambda` is dimensionless and the ratio d/C_D makes the decay
   scale-free in crown size, so a wide-crowned species cools further.

2. Crown competition: where crowns overlap, accumulated crown projection area
   (CCA) suppresses delivered cooling through a logistic damping term,

       f(CCA) = 1 / (1 + exp( K * (CCA - CCA_threshold) ))

   so cooling saturates rather than summing without limit.

3. Contributions are summed across trees. The result is a value per fine grid
   point, ordered x-major over `fine_grid_points` -- see the index-conventions
   note in `grid.py`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from . import config
from .species import TreeSpecies


class CoolingModel:
    """Radial cooling with competition damping, over a fine evaluation grid."""

    def __init__(self, decay_lambda=None, cca_threshold=None,
                 competition_k=None, shade_weight=None, evap_weight=None):
        self.decay_lambda = (decay_lambda if decay_lambda is not None
                             else config.COOLING_PARAMS["decay_lambda"])
        self.cca_threshold = (cca_threshold if cca_threshold is not None
                              else config.COOLING_PARAMS["cca_threshold"])
        self.competition_k = (competition_k if competition_k is not None
                              else config.COOLING_PARAMS["competition_k"])
        self.shade_weight = shade_weight if shade_weight is not None else config.SHADE_WEIGHT
        self.evap_weight = evap_weight if evap_weight is not None else config.EVAP_WEIGHT

        # Constructed then reassigned, matching the reference implementation:
        # TreeSpecies() takes the default split, then the model's own weights
        # are written onto it. Constructing TreeSpecies also rewrites every
        # species' CPA in the shared class dict -- see species.py.
        self.tree_species = TreeSpecies()
        self.tree_species.shade_weight = self.shade_weight
        self.tree_species.evap_weight = self.evap_weight

    def calculate_cooling_contribution(self, tree_pos, tree_species_name,
                                       fine_grid_points, cca_values=None):
        """One tree's cooling field, optionally damped by competition."""
        species_params = self.tree_species.get_species_params(tree_species_name)
        if not species_params:
            return np.zeros(len(fine_grid_points))

        D_j = self.tree_species.get_normalized_cooling_potential(tree_species_name)
        distances = cdist([tree_pos], fine_grid_points, "euclidean")[0]
        crown_diameter = species_params["crown_diameter_m"]

        decay_factor = np.exp(-self.decay_lambda * (distances ** 2) /
                              (crown_diameter ** 2))

        cooling = D_j * decay_factor

        if cca_values is not None:
            # Where crowns pile up, K*(CCA - threshold) is large and exp()
            # overflows to +inf, giving a damping factor of exactly 0.0 -- which
            # is the correct limit, so the overflow is harmless and the result
            # is bit-identical to the reference implementation. The warning is
            # silenced only for this expression rather than process-wide, so a
            # genuine numerical problem elsewhere still surfaces.
            with np.errstate(over="ignore"):
                competition_factor = 1 / (1 + np.exp(
                    self.competition_k * (cca_values - self.cca_threshold)
                ))
            cooling *= competition_factor

        return cooling

    def calculate_cca(self, tree_placements, tree_species_list, fine_grid_points):
        """Accumulated crown projection area at each fine grid point.

        A tree contributes its full CPA to every point inside its crown radius
        and nothing outside -- a hard disc, not a taper.
        """
        cca_values = np.zeros(len(fine_grid_points))

        for pos, species in zip(tree_placements, tree_species_list):
            species_params = self.tree_species.get_species_params(species)
            if not species_params:
                continue
            crown_radius = species_params["crown_diameter_m"] / 2
            distances = cdist([pos], fine_grid_points, "euclidean")[0]
            within_crown = distances <= crown_radius
            cpa = species_params["CPA"]
            cca_values[within_crown] += cpa

        return cca_values

    def calculate_total_cooling(self, tree_placements, tree_species_list,
                                fine_grid_points, apply_competition=True):
        """Summed cooling field and the CCA field behind it.

        Returns `(total_cooling, cca_values)`; `cca_values` is None when
        competition is disabled.
        """
        if apply_competition:
            cca_values = self.calculate_cca(tree_placements, tree_species_list,
                                            fine_grid_points)
        else:
            cca_values = None

        total_cooling = np.zeros(len(fine_grid_points))

        for pos, species in zip(tree_placements, tree_species_list):
            cooling = self.calculate_cooling_contribution(
                pos, species, fine_grid_points, cca_values
            )
            total_cooling += cooling

        return total_cooling, cca_values

    def decay_curve(self, species_name, max_distance_m=50.0, n_points=500):
        """Single-tree decay profile, for plotting. Consumes no randomness."""
        distances = np.linspace(0, max_distance_m, n_points)
        params = self.tree_species.get_species_params(species_name)
        if not params:
            return distances, np.zeros_like(distances)
        D_j = self.tree_species.get_normalized_cooling_potential(species_name)
        crown_diameter = params["crown_diameter_m"]
        cooling = D_j * np.exp(-self.decay_lambda * (distances ** 2) /
                               (crown_diameter ** 2))
        return distances, cooling


# The reference implementation's name for this class, kept as an alias so that
# code and documentation written against `AuditedCode_1.py` still reads across.
CorrectedCoolingModel = CoolingModel
