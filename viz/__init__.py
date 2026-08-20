"""Visualization: one theme, one figure module, no styling anywhere else.

`theme` holds the palette, typography, and the Plotly template that every
figure inherits. `figures` holds one function per chart. Nothing in this
package computes science or consumes randomness -- see `figures.py`.
"""
from __future__ import annotations

from . import export, figures, theme
from .theme import register

__all__ = ["export", "figures", "theme", "register"]
