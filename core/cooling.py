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

2. Crown competition: where crowns overlap, the accumulated crown overlap (CCA)
   suppresses delivered cooling through a logistic damping term,

       f(CCA) = 1 / (1 + exp( K * (CCA - CCA_threshold) ))

   so cooling saturates rather than summing without limit.

   **CCA is a dimensionless count of overlapping crowns**, and a tree is
   exempt from its own crown. Both of those are the substance of the model
   rather than bookkeeping, so they are worth stating plainly:

     0 other crowns over a cell -> damping 0.9975  (essentially undamped)
     1 other crown              -> 0.7311
     2 other crowns             -> 0.0180
     3 other crowns             -> 1.2e-04

   It counts crowns rather than summing their areas because `CCA_threshold`
   is 1.2 and the sweep range for it is [0.5, 2.0]. Against a crown count
   those mean "competition begins just past single-layer cover"; against an
   accumulated area they mean nothing, since one crown alone is 70.9-452.4 m².
   An earlier formulation did accumulate area, which drove the logistic far
   past saturation and made it a step function at the crown edge: cooling was
   exactly zero beneath every canopy -- 16.3% of the domain and 46.3% of all
   generated cooling -- while a cell one metre outside the drip line got
   99.75% of it. That is backwards, since canopy shade is greatest directly
   under a tree. Corrected by research-lead decision; the corresponding
   discussion is in `docs/DECISIONS.md`.

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
            # `cca_values` here is the count of OTHER crowns over each cell --
            # this tree's own is already subtracted by `calculate_total_cooling`.
            #
            # The overflow guard is kept although it is no longer reachable. A
            # crown count cannot exceed the number of trees placed, so the
            # exponent stays small; it was accumulated crown AREA that used to
            # send exp() to +inf. Suppression is scoped to this one expression
            # rather than process-wide, so a genuine numerical problem elsewhere
            # still surfaces, and it changes no value either way.
            with np.errstate(over="ignore"):
                competition_factor = 1 / (1 + np.exp(
                    self.competition_k * (cca_values - self.cca_threshold)
                ))
            cooling *= competition_factor

        return cooling

    def calculate_cca(self, tree_placements, tree_species_list, fine_grid_points):
        """How many crowns cover each fine grid point. Dimensionless.

        Each covering crown contributes 1.0; a tree covers a point or it does
        not, a hard disc rather than a taper.

        This counts crowns rather than summing their areas, which is what makes
        `cca_threshold = 1.2` mean anything. See the module docstring.
        """
        cca_values = np.zeros(len(fine_grid_points))

        for pos, species in zip(tree_placements, tree_species_list):
            species_params = self.tree_species.get_species_params(species)
            if not species_params:
                continue
            crown_radius = species_params["crown_diameter_m"] / 2
            distances = cdist([pos], fine_grid_points, "euclidean")[0]
            within_crown = distances <= crown_radius
            cca_values[within_crown] += 1.0

        return cca_values

    def calculate_own_crown_mask(self, tree_pos, tree_species_name,
                                 fine_grid_points):
        """Boolean mask of the fine cells inside THIS tree's own crown.

        Used to remove a tree's own crown from the overlap count it is damped
        by, so that it never competes with itself.
        """
        species_params = self.tree_species.get_species_params(tree_species_name)
        if not species_params:
            return np.zeros(len(fine_grid_points), dtype=bool)
        crown_radius = species_params["crown_diameter_m"] / 2
        distances = cdist([tree_pos], fine_grid_points, "euclidean")[0]
        return distances <= crown_radius

    def calculate_total_cooling(self, tree_placements, tree_species_list,
                                fine_grid_points, apply_competition=True):
        """Summed cooling field and the crown-overlap field behind it.

        Returns `(total_cooling, cca_values)`; `cca_values` is None when
        competition is disabled.

        Each tree is damped by the count of *other* crowns over each cell, never
        by its own -- `calculate_cooling_contribution`'s signature is unchanged
        and the subtraction happens here, because this is the single entry point
        every consumer of the model reaches.
        """
        if apply_competition:
            cca_values = self.calculate_cca(tree_placements, tree_species_list,
                                            fine_grid_points)
        else:
            cca_values = None

        total_cooling = np.zeros(len(fine_grid_points))

        for pos, species in zip(tree_placements, tree_species_list):
            if cca_values is not None:
                own_mask = self.calculate_own_crown_mask(
                    pos, species, fine_grid_points
                )
                cca_for_this_tree = cca_values - own_mask
            else:
                cca_for_this_tree = None
            cooling = self.calculate_cooling_contribution(
                pos, species, fine_grid_points, cca_for_this_tree
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
