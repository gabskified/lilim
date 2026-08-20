"""Production configuration — the single source of truth for every constant.

Every value here is transcribed from `legacy/AuditedCode_1.py`'s
`main_revised_validation()` (:3380-3404) and cross-checked against
`tools/morris/secpi_factors.py` (:64-80), which holds the same block for the
Morris harness. The two agreed exactly at transcription time; the parity check
re-verifies the numbers that matter by execution rather than by inspection.

Nothing in this module is tunable at import time. The interactive workbench
passes overrides explicitly, and shows the user when their settings have
departed from the production values below.
"""
from __future__ import annotations

# ---------------------------------------------------------------- grid geometry
# Coarse cells are the planting/optimization units; fine cells are where cooling
# is evaluated. 10 x 10 coarse cells of 10 m = a 100 x 100 m synthetic domain,
# discretised for evaluation at 1 m.
COARSE_GRID = {"width": 10, "height": 10, "cell_size": 10.0}
FINE_GRID = {"cell_size": 1.0}

# ---------------------------------------------------------------- land-use codes
# These integer codes are the on-grid encoding used throughout. They are not
# arbitrary labels -- the CA writes them directly into the coarse grid array.
P_CODE = 1   # Prohibited  -- already occupied/built, unavailable for planting
A_CODE = 3   # Available   -- open and plantable
V_CODE = 4   # Vulnerable  -- equity-weighted subset requiring priority cooling

# Target composition bands the CA calibrates itself into on every run.
P_TARGET_RANGE = (0.55, 0.65)
V_TARGET_RANGE = (0.05, 0.10)

# ---------------------------------------------------------------- CA parameters
# p0 is fixed at 1.0 and is no longer independently calibrated: p0 and gamma are
# both multiplicative scale factors in the first update, p(1) = gamma*omega*p0,
# so they are not separately identifiable. gamma alone carries the growth-rate
# calibration. See docs/DECISIONS.md D-10.
CA_PARAMS = {
    "morphology": "organic",
    "p_init": 0.15,
    "gamma": 4.0,
    "p0": 1.0,
    "theta": 3,
}

MORPHOLOGIES = ("organic", "linear")

# ---------------------------------------------------------------- cooling model
# These are the actual production values, not illustrative ones.
COOLING_PARAMS = {
    "decay_lambda": 1.9,
    "cca_threshold": 1.2,
    "competition_k": 5.0,
}

# Split of the normalized cooling potential D_j between crown-area (shade) and
# leaf-area (evapotranspiration) terms.
SHADE_WEIGHT = 0.7
EVAP_WEIGHT = 0.3

# ---------------------------------------------------------------- optimizer
ACO_PARAMS = {
    "n_trees": 5,
    "n_ants": 20,
    "n_iterations": 40,
    "evaporation_rate": 0.5,
    "alpha": 1.0,
    "beta": 2.0,
    "q0": 0.7,
}

# ---------------------------------------------------------------- normalization
# Fixed study-wide reference cutoffs (Option B). Calibrated ONCE per study by
# Monte Carlo pooling of random valid placements, then reused identically by
# every optimizer instance so that SECPI stays comparable across analyses.
CUTOFF_CALIB = {
    "n_trees_range": (1, 6),
    "n_samples": 100,
    "random_seed": 42,
}

# ---------------------------------------------------------------- study defaults
# The canonical grid seed used throughout the regenerated results.
DEFAULT_GRID_SEED = 42

# Restart count per configuration, and the tree-count sweep axis.
N_RUNS = 5
K_VALUES = [1, 2, 3, 4, 5, 6]

# Seed bases for the scenario sweeps. The SAME seed is used for restart index
# `run` in both the WITH and WITHOUT arms at the same k, so a pair differs only
# by the equity manipulation and not also by its pheromone initialisation.
SEED_BASE_KSWEEP = 4_000_000
SEED_BASE_SUBSET = 3_000_000

# Palette size is the sweep axis for the species-subset analysis; tree count is
# held at five there.
SUBSET_N_TREES = 5

# ---------------------------------------------------------------- equity
# The three discrete weight levels the implementation actually applies, by
# distance from the nearest Vulnerable cell.
EQUITY_LEVELS = (1.0, 1.5, 2.0)
EQUITY_NEAR_RADIUS_M = 10.0   # <= this distance  -> 2.0
EQUITY_MID_RADIUS_M = 20.0    # <= this distance  -> 1.5


def kseed(k: int, run: int) -> int:
    """Seed for restart `run` of the k-sweep at tree count `k`."""
    return SEED_BASE_KSWEEP + k * 100 + run


def sseed(subset_id: int, run: int) -> int:
    """Seed for restart `run` of species subset `subset_id`."""
    return SEED_BASE_SUBSET + subset_id * 100 + run


def production_config() -> dict:
    """The full production configuration as one nested dict."""
    return {
        "coarse_grid": dict(COARSE_GRID),
        "fine_grid": dict(FINE_GRID),
        "ca_params": dict(CA_PARAMS),
        "cooling_params": dict(COOLING_PARAMS),
        "aco_params": dict(ACO_PARAMS),
        "cutoff_calib": dict(CUTOFF_CALIB),
        "grid_seed": DEFAULT_GRID_SEED,
    }


def departures_from_production(config: dict) -> list[str]:
    """Human-readable list of where `config` differs from production values.

    Used by the workbench to tell the user, plainly, when what they are looking
    at is no longer the configuration behind the reported results.
    """
    base = production_config()
    out: list[str] = []
    for section, values in base.items():
        if not isinstance(values, dict):
            if config.get(section) != values:
                out.append(f"{section}: {config.get(section)} (production {values})")
            continue
        given = config.get(section, {})
        for key, expected in values.items():
            actual = given.get(key, expected)
            if actual != expected:
                out.append(f"{section}.{key}: {actual} (production {expected})")
    return out
