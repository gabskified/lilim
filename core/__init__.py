"""SECPI reference implementation — the science, with no interface attached.

Import order matters only in that `config` has no dependencies and everything
else reads from it. There is no hidden global state in this package beyond
`TreeSpecies.SPECIES_DATA` (documented in `species.py`) and numpy's global
random state, which every entry point seeds explicitly before use.

Typical use:

    from lilim.core import build_context, run_once

    ctx = build_context(grid_seed=42)
    aco, result = run_once(ctx.grid, ctx.cooling_model, ctx.cutoffs, seed=4000105)
    print(result["raw_secpi"], result["history_best"][-1])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config, cooling, grid, scenarios, secpi, sensitivity, species
from .aco import AntColonySystemACO, run_once
from .cooling import CoolingModel
from .grid import TwoLevelUrbanGrid, build_grid, without_equity_arm
from .scenarios import ScenarioContext, run_k_sweep, run_subset_sweep, species_subsets
from .secpi import calibrate_reference_cutoffs, normalize
from .sensitivity import SensitivityAnalyzer
from .species import TreeSpecies

__all__ = [
    "config", "cooling", "grid", "scenarios", "secpi", "sensitivity", "species",
    "AntColonySystemACO", "run_once", "CoolingModel", "TwoLevelUrbanGrid",
    "build_grid", "without_equity_arm", "ScenarioContext", "run_k_sweep",
    "run_subset_sweep", "species_subsets", "calibrate_reference_cutoffs",
    "normalize", "SensitivityAnalyzer", "TreeSpecies", "StudyContext",
    "build_context",
]


@dataclass
class StudyContext:
    """A generated grid, its cooling model, and the study-wide cutoffs.

    Everything downstream needs these three together, and the cutoffs must be
    the SAME object across every analysis in a run or the scores stop being
    comparable. Bundling them makes that hard to get wrong by accident.
    """
    grid: Any
    cooling_model: Any
    cutoffs: tuple
    grid_seed: int
    settings: dict = field(default_factory=dict)

    @property
    def species_list(self):
        return self.cooling_model.tree_species.species_list

    def composition(self):
        return self.grid.composition()

    def scenario_context(self):
        return ScenarioContext(self.grid, self.cooling_model, self.cutoffs)


def build_context(grid_seed: int = config.DEFAULT_GRID_SEED,
                  ca_params: dict | None = None,
                  morphology: str | None = None,
                  cooling_params: dict | None = None,
                  cutoff_calib: dict | None = None,
                  coarse_grid: dict | None = None,
                  fine_grid: dict | None = None) -> StudyContext:
    """Build a grid, a cooling model, and the study-wide reference cutoffs.

    This reproduces the reference implementation's setup exactly: seed the
    global RNG, generate the grid, construct the cooling model, then calibrate
    the cutoffs ONCE with their own fixed seed. Every optimizer run in the
    resulting context must be given these same cutoffs.
    """
    ca = dict(config.CA_PARAMS)
    if ca_params:
        ca.update(ca_params)
    if morphology:
        ca["morphology"] = morphology

    cool = dict(config.COOLING_PARAMS)
    if cooling_params:
        cool.update(cooling_params)

    calib = dict(config.CUTOFF_CALIB)
    if cutoff_calib:
        calib.update(cutoff_calib)

    g = build_grid(grid_seed=grid_seed, ca_params=ca,
                   morphology=ca["morphology"],
                   coarse_grid=coarse_grid, fine_grid=fine_grid)

    model = CoolingModel(
        decay_lambda=cool["decay_lambda"],
        cca_threshold=cool["cca_threshold"],
        competition_k=cool["competition_k"],
    )

    cutoffs = calibrate_reference_cutoffs(
        g, model, model.tree_species.species_list,
        n_trees_range=calib["n_trees_range"],
        n_samples=calib["n_samples"],
        random_seed=calib["random_seed"],
    )

    return StudyContext(
        grid=g, cooling_model=model, cutoffs=cutoffs, grid_seed=grid_seed,
        settings={"ca_params": ca, "cooling_params": cool, "cutoff_calib": calib},
    )
