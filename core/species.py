"""The six Philippine Tree Functional Types and their biophysical parameters.

Ported from `legacy/AuditedCode_1.py:1613-1747` with the numbers unchanged.

TWO THINGS HERE ARE DELIBERATELY NOT MODERNISED
-----------------------------------------------
1. `SPECIES_DATA` is a CLASS attribute, mutated in place. Constructing a
   `TreeSpecies` writes each species' `CPA` back into it, and the sensitivity
   sweep writes parameter perturbations into it. Making it an instance
   attribute would be cleaner and would change results -- the sweep's
   snapshot/restore discipline (see `sensitivity.py`) exists precisely because
   this is shared global state. Leave it alone.

2. `LAI` is a hardcoded, adopted value. It is NOT derived from the `l0/h0/l1/h1`
   allometric constants sitting beside it in the same table, even though those
   constants describe a complete DBH -> LAI pipeline. `get_computed_lai()`
   implements that pipeline and returns values roughly two orders of magnitude
   smaller (~0.01-0.09 against the adopted ~3.15-6.07). The main analysis uses
   the adopted values; the allometric pipeline is exercised only by the
   sensitivity sweep, and there only through *relative* change. See
   docs/DECISIONS.md D-09.
"""
from __future__ import annotations

import numpy as np

from . import config


class TreeSpecies:
    """Six tree functional types with crown geometry, leaf area, and colour."""

    # h0, h1, l0, l1 are the manuscript's own Table 4 constants (author-team
    # estimates), per DBH = (h/h0)^(1/h1) and LAI = l0 * DBH^l1.
    #
    # Confirmed binomials: Narra = Pterocarpus indicus;
    # Akleng-parang = Albizia lebbeck (author-corrected).
    SPECIES_DATA = {
        "Narra": {
            "crown_diameter_m": 23.0,
            "height_m": 19.5,
            "LAI": 6.07,
            "l0": 0.25, "l1": 1.9, "h0": 51.2, "h1": 0.75,
            "color": "#E74C3C",
        },
        "Talisay": {
            "crown_diameter_m": 12.0,
            "height_m": 14.0,
            "LAI": 4.40,
            "l0": 0.20, "l1": 1.8, "h0": 47.2, "h1": 0.71,
            "color": "#F39C12",
        },
        "Banaba": {
            "crown_diameter_m": 11.0,
            "height_m": 11.0,
            "LAI": 3.87,
            "l0": 0.20, "l1": 1.8, "h0": 45.8, "h1": 0.72,
            "color": "#9B59B6",
        },
        "Kabiki": {
            "crown_diameter_m": 11.0,
            "height_m": 15.0,
            "LAI": 4.12,
            "l0": 0.22, "l1": 1.85, "h0": 48.5, "h1": 0.73,
            "color": "#3498DB",
        },
        "Duhat": {
            "crown_diameter_m": 9.5,
            "height_m": 20.0,
            "LAI": 3.52,
            "l0": 0.18, "l1": 1.75, "h0": 42.3, "h1": 0.70,
            "color": "#1ABC9C",
        },
        "Akleng-parang": {
            "crown_diameter_m": 24.0,
            "height_m": 17.5,
            "LAI": 3.15,
            "l0": 0.15, "l1": 1.65, "h0": 46.1, "h1": 0.68,
            "color": "#E91E63",
        },
    }

    # Scientific binomials, for display only. Nothing computes from these.
    BINOMIALS = {
        "Narra": "Pterocarpus indicus",
        "Talisay": "Terminalia catappa",
        "Banaba": "Lagerstroemia speciosa",
        "Kabiki": "Mimusops elengi",
        "Duhat": "Syzygium cumini",
        "Akleng-parang": "Albizia lebbeck",
    }

    def __init__(self, shade_weight: float = config.SHADE_WEIGHT,
                 evap_weight: float = config.EVAP_WEIGHT):
        self.species_list = list(self.SPECIES_DATA.keys())
        self.shade_weight = shade_weight
        self.evap_weight = evap_weight
        self._calculate_cpa_and_normalize()

    def _calculate_cpa_and_normalize(self):
        """Write each species' crown projection area, and cache the maxima.

        `max_CPA` and `max_LAI` are INSTANCE attributes and are live
        denominators in `get_normalized_cooling_potential`. Any code that
        perturbs `SPECIES_DATA` must call this again -- and any code that
        restores `SPECIES_DATA` must restore these two as well, or the
        denominators stay contaminated.
        """
        all_cpa = []
        all_lai = []
        for species in self.species_list:
            data = self.SPECIES_DATA[species]
            data["CPA"] = (np.pi / 4) * (data["crown_diameter_m"] ** 2)
            all_cpa.append(data["CPA"])
            all_lai.append(data["LAI"])

        self.max_CPA = max(all_cpa)
        self.max_LAI = max(all_lai)

    def get_normalized_cooling_potential(self, species_name: str) -> float:
        """D_j -- the species' cooling potential, normalized to the palette.

            D_j = shade_weight * (CPA_j / max CPA) + evap_weight * (LAI_j / max LAI)
        """
        data = self.get_species_params(species_name)
        if not data:
            return 0
        cpa_norm = data["CPA"] / self.max_CPA
        lai_norm = data["LAI"] / self.max_LAI
        return self.shade_weight * cpa_norm + self.evap_weight * lai_norm

    def get_species_params(self, species_name: str) -> dict:
        """The species' live parameter dict. Mutating the result mutates state."""
        return self.SPECIES_DATA.get(species_name, {})

    def get_crown_radius(self, species_name: str) -> float:
        data = self.get_species_params(species_name)
        return data.get("crown_diameter_m", 0) / 2

    def get_species_color(self, species_name: str) -> str:
        data = self.get_species_params(species_name)
        return data.get("color", "gray")

    def get_binomial(self, species_name: str) -> str:
        return self.BINOMIALS.get(species_name, "")

    def get_dbh(self, species_name, height_override=None,
                h0_override=None, h1_override=None) -> float:
        """DBH from height, inverting the power-form allometry h = h0 * DBH^h1:

            DBH = (h / h0) ** (1 / h1)

        Overrides let the sensitivity sweep perturb h, h0, h1 independently
        without mutating SPECIES_DATA.
        """
        data = self.get_species_params(species_name)
        if not data:
            return 0.0
        h = height_override if height_override is not None else data["height_m"]
        h0 = h0_override if h0_override is not None else data["h0"]
        h1 = h1_override if h1_override is not None else data["h1"]
        if h <= 0 or h0 <= 0 or h1 == 0:
            return 0.0
        return (h / h0) ** (1.0 / h1)

    def get_computed_lai(self, species_name, height_override=None,
                         l0_override=None, l1_override=None,
                         h0_override=None, h1_override=None) -> float:
        """LAI from the allometric pipeline: LAI = l0 * DBH^l1.

        This is the *model-consistent* LAI, distinct from the adopted
        `SPECIES_DATA['LAI']` the main analysis uses. See the module docstring.
        """
        data = self.get_species_params(species_name)
        if not data:
            return 0.0
        dbh = self.get_dbh(species_name, height_override, h0_override, h1_override)
        l0 = l0_override if l0_override is not None else data["l0"]
        l1 = l1_override if l1_override is not None else data["l1"]
        return l0 * (dbh ** l1)

    def summary_rows(self) -> list[dict]:
        """One row per species, for display. Derived, never authoritative."""
        return [
            {
                "species": name,
                "binomial": self.get_binomial(name),
                "crown_diameter_m": self.SPECIES_DATA[name]["crown_diameter_m"],
                "height_m": self.SPECIES_DATA[name]["height_m"],
                "CPA_m2": self.SPECIES_DATA[name]["CPA"],
                "LAI_adopted": self.SPECIES_DATA[name]["LAI"],
                "LAI_computed": self.get_computed_lai(name),
                "DBH_m": self.get_dbh(name),
                "D_j": self.get_normalized_cooling_potential(name),
                "color": self.SPECIES_DATA[name]["color"],
            }
            for name in self.species_list
        ]
