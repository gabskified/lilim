"""One-at-a-time sensitivity analysis over the 40 model parameters.

Ported from `legacy/AuditedCode_1.py:766-1105`.

WHAT IT SWEEPS
--------------
Four categories, 40 factors total:
  Cooling_Model      3  -- decay_lambda, cca_threshold, competition_k
  Weighting          1  -- shade_weight (evap_weight is its complement)
  Species_Morphology 12 -- crown_diameter_m and height_m, per species
  Species_Allometry  24 -- l0, l1, h0, h1, per species

Each factor is evaluated at both ends of its range with everything else held at
baseline, and its sensitivity index is the absolute change in mean SECPI
divided by the baseline SECPI. Morphology and cooling factors sweep to fixed
bounds; species factors sweep +/-20% of their own base value.

⚠️  EVERY EVALUATION IS A FULL PRODUCTION OPTIMIZER RUN. The default sweep is
3 baseline runs + 40 factors x 2 bounds x n_samples. At n_samples=3 that is 243
optimizer runs, which is a batch job measured in tens of minutes, not a click.

STATE ISOLATION — THE PART THAT LOOKS PARANOID AND IS NOT
----------------------------------------------------------
`TreeSpecies.SPECIES_DATA` is a CLASS attribute shared by every instance in the
process, so species perturbations are writes to global state. Without the
snapshot/restore below: the LAI write compounds geometrically across repeated
evaluations so the routine is not idempotent; every evaluation after the first
species perturbation runs against contaminated data, making "all other
parameters held at baseline" false -- which is the entire definition of a
one-at-a-time sweep; and the contamination escapes into any later step in the
same process.

THREE things are snapshotted, not one. `_calculate_cpa_and_normalize()` caches
`max_CPA` and `max_LAI` as INSTANCE attributes and
`get_normalized_cooling_potential()` divides by both -- they are live
denominators. Restoring only the dict would leave contaminated denominators
behind. They are restored from the snapshot directly rather than recomputed, so
the restored values are bit-identical by construction.

The dict is restored IN PLACE at both levels -- outer mapping and each species'
inner mapping -- rather than by rebinding, so any code holding a reference (for
example via `get_species_params()`, which returns the inner dict itself) sees
the restored values.

The snapshot is taken before ANY write point, including the cooling-model
construction (which builds a `TreeSpecies` and so rewrites every CPA), and is
restored in an unconditional `finally`.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from . import config
from .aco import AntColonySystemACO
from .cooling import CoolingModel
from .species import TreeSpecies

CATEGORY_ORDER = ("Cooling_Model", "Weighting", "Species_Morphology", "Species_Allometry")

# The nominal vector every cooling/weighting evaluation starts from.
BASELINE_COOLING_VECTOR = {
    "decay_lambda": config.COOLING_PARAMS["decay_lambda"],
    "cca_threshold": config.COOLING_PARAMS["cca_threshold"],
    "competition_k": config.COOLING_PARAMS["competition_k"],
    "shade_weight": config.SHADE_WEIGHT,
}


def define_parameters() -> dict:
    """The 40 swept factors, with base values and ranges, in category order."""
    params: dict = {
        "Cooling_Model": {
            "decay_lambda": {"base": 1.9, "range": [0.5, 3.0]},
            "cca_threshold": {"base": 1.2, "range": [0.5, 2.0]},
            "competition_k": {"base": 5.0, "range": [1.0, 10.0]},
        },
        "Weighting": {
            "shade_weight": {"base": 0.7, "range": [0.5, 0.9]},
        },
        "Species_Morphology": {},
        "Species_Allometry": {},
    }

    tree_species = TreeSpecies()
    for species in tree_species.species_list:
        sp = tree_species.get_species_params(species)

        base_cd = sp["crown_diameter_m"]
        params["Species_Morphology"][f"{species}.crown_diameter_m"] = {
            "base": base_cd, "range": [base_cd * 0.8, base_cd * 1.2],
            "species": species, "param_name": "crown_diameter_m",
        }

        base_h = sp["height_m"]
        params["Species_Morphology"][f"{species}.height_m"] = {
            "base": base_h, "range": [base_h * 0.8, base_h * 1.2],
            "species": species, "param_name": "height_m",
        }

        # Allometric constants: the published table's own values, perturbed
        # +/-20% for consistency with crown diameter and height above.
        for pname in ("l0", "l1", "h0", "h1"):
            base_val = sp[pname]
            params["Species_Allometry"][f"{species}.{pname}"] = {
                "base": base_val, "range": [base_val * 0.8, base_val * 1.2],
                "species": species, "param_name": pname,
            }

    return params


class SensitivityAnalyzer:
    """OAT sweep over the 40 factors, at production optimizer settings."""

    def __init__(self, base_grid, base_aco_config=None, reference_cutoffs=None):
        self.base_grid = base_grid
        self.base_aco_config = dict(base_aco_config or config.ACO_PARAMS)
        self.reference_cutoffs = reference_cutoffs
        self.parameter_definitions = define_parameters()
        self.results: list[dict] = []
        self.baseline_secpi = None

    # ------------------------------------------------------- one evaluation
    def run_single_evaluation(self, cooling_params, species_modifications=None):
        """One optimizer run under a perturbed configuration.

        See the module docstring for why the snapshot/restore is structured the
        way it is. Do not simplify it.
        """
        species_data_snapshot = copy.deepcopy(TreeSpecies.SPECIES_DATA)
        normalization_snapshot = None

        try:
            cooling_model = CoolingModel(
                decay_lambda=cooling_params.get("decay_lambda", 1.9),
                cca_threshold=cooling_params.get("cca_threshold", 1.2),
                competition_k=cooling_params.get("competition_k", 5.0),
                shade_weight=cooling_params.get("shade_weight", 0.7),
                evap_weight=1.0 - cooling_params.get("shade_weight", 0.7),
            )

            if species_modifications:
                ts = cooling_model.tree_species
                # Capture the denominators of the very instance that will run
                # _calculate_cpa_and_normalize() below, before any perturbation.
                normalization_snapshot = (ts, ts.max_CPA, ts.max_LAI)
                for species, param_name, value in species_modifications:
                    if species not in ts.SPECIES_DATA:
                        continue
                    if param_name in ("l0", "l1", "h0", "h1"):
                        # Perturb the constant, recompute LAI through the real
                        # allometric pipeline, then apply the RELATIVE change to
                        # the adopted LAI baseline. Using the absolute computed
                        # LAI instead would let the two-orders-of-magnitude gap
                        # between computed (~0.01-0.09) and adopted (~3.15-6.07)
                        # swamp the measurement.
                        baseline_computed_lai = ts.get_computed_lai(species)
                        perturbed_computed_lai = ts.get_computed_lai(
                            species, **{f"{param_name}_override": value}
                        )
                        ratio = (perturbed_computed_lai / baseline_computed_lai
                                 if baseline_computed_lai > 0 else 1.0)
                        hardcoded_lai = ts.SPECIES_DATA[species]["LAI"]
                        ts.SPECIES_DATA[species]["LAI"] = hardcoded_lai * ratio
                    else:
                        ts.SPECIES_DATA[species][param_name] = value
                        if param_name == "crown_diameter_m":
                            ts.SPECIES_DATA[species]["CPA"] = (np.pi / 4) * (value ** 2)
                ts._calculate_cpa_and_normalize()

            # Production optimizer settings, so sensitivity indices are computed
            # on the same-fidelity search as the headline results.
            aco = AntColonySystemACO(
                self.base_grid, cooling_model,
                n_trees=self.base_aco_config["n_trees"],
                n_ants=self.base_aco_config["n_ants"],
                n_iterations=self.base_aco_config["n_iterations"],
                evaporation_rate=self.base_aco_config["evaporation_rate"],
                alpha=self.base_aco_config["alpha"],
                beta=self.base_aco_config["beta"],
                q0=self.base_aco_config["q0"],
                reference_cutoffs=self.reference_cutoffs,
            )
            aco.run()
            return aco.best_secpi if aco.best_secpi else 0

        except Exception:
            return 0

        finally:
            # Restore in place at both levels, so held references see it.
            live_species_data = TreeSpecies.SPECIES_DATA
            for species_name in list(live_species_data.keys()):
                if species_name not in species_data_snapshot:
                    del live_species_data[species_name]
            for species_name, saved_fields in species_data_snapshot.items():
                live_fields = live_species_data.get(species_name)
                if live_fields is None:
                    live_species_data[species_name] = copy.deepcopy(saved_fields)
                else:
                    live_fields.clear()
                    live_fields.update(copy.deepcopy(saved_fields))

            # Restore the cached denominators from the snapshot rather than
            # recomputing, so they are bit-identical.
            if normalization_snapshot is not None:
                snapshot_ts, saved_max_cpa, saved_max_lai = normalization_snapshot
                snapshot_ts.max_CPA = saved_max_cpa
                snapshot_ts.max_LAI = saved_max_lai

    # ------------------------------------------------------------ the sweep
    def run_oat_analysis(self, n_samples=3,
                         progress: Callable[[int, int, str], None] | None = None):
        """Sweep every factor to both bounds. Returns a list of result rows.

        `progress(done, total, label)` is called per factor. It must not
        consume randomness.
        """
        baseline_scores = []
        for _ in range(n_samples):
            baseline_scores.append(
                self.run_single_evaluation(dict(BASELINE_COOLING_VECTOR)))
        self.baseline_secpi = float(np.mean(baseline_scores))

        self.results = []
        total_params = sum(len(p) for p in self.parameter_definitions.values())
        done = 0

        for category, params in self.parameter_definitions.items():
            for param_name, param_info in params.items():
                low_val, high_val = param_info["range"]
                low_scores, high_scores = [], []

                for _ in range(n_samples):
                    if category in ("Cooling_Model", "Weighting"):
                        base_vec = dict(BASELINE_COOLING_VECTOR)

                        low_vec = base_vec.copy()
                        low_vec[param_name] = low_val
                        low_scores.append(self.run_single_evaluation(low_vec))

                        high_vec = base_vec.copy()
                        high_vec[param_name] = high_val
                        high_scores.append(self.run_single_evaluation(high_vec))
                    else:
                        species = param_info.get("species")
                        actual_param = param_info.get("param_name")
                        low_scores.append(self.run_single_evaluation(
                            {}, [(species, actual_param, low_val)]))
                        high_scores.append(self.run_single_evaluation(
                            {}, [(species, actual_param, high_val)]))

                secpi_low = float(np.mean(low_scores))
                secpi_high = float(np.mean(high_scores))
                absolute_effect = abs(secpi_high - secpi_low)
                sensitivity_index = (absolute_effect / self.baseline_secpi
                                     if self.baseline_secpi > 0 else 0.0)

                self.results.append({
                    "parameter": param_name,
                    "category": category,
                    "secpi_low": secpi_low,
                    "secpi_high": secpi_high,
                    "absolute_effect": absolute_effect,
                    "sensitivity_index": float(sensitivity_index),
                })

                done += 1
                if progress:
                    progress(done, total_params, f"{category}: {param_name}")

        # Left in definition order deliberately -- that order is what makes the
        # sweep comparable factor-by-factor against the reference
        # implementation. Use `ranked()` for anything that wants it sorted.
        return self.results

    def ranked(self, top=None) -> list[dict]:
        """Results sorted by sensitivity index, most sensitive first."""
        rows = sorted(self.results, key=lambda r: r["sensitivity_index"], reverse=True)
        return rows[:top] if top else rows

    def category_totals(self) -> dict:
        """Summed sensitivity index per category."""
        out: dict = {}
        for row in self.results:
            out[row["category"]] = out.get(row["category"], 0.0) + row["sensitivity_index"]
        return out
