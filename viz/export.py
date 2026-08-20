"""Saving figures to disk, with their caveat attached.

The app shows captions as page text, because that is the only way they stay
aligned at any window size (see the note in `figures.py`). But a PNG dropped
into a document or a slide leaves that text behind, and a figure separated from
its caveat is exactly how a qualified result becomes an unqualified one. So the
export path re-attaches the caption to the figure itself.

Two formats:

* **PNG** at 3x scale, roughly 300 dpi, matching the resolution the reference
  implementation saved its figures at. Needs `kaleido`, which drives a
  Chrome-family browser to do the rendering.
* **HTML**, self-contained and still interactive. Pure Python, no external
  binary, so it is *always* available. This is the fallback when kaleido cannot
  render -- and for a handoff tool it is arguably the better artifact anyway,
  since the recipient can zoom and hover rather than squint at a raster.

`available()` probes rather than assumes. kaleido needs a browser it can drive,
and whether one is present is a property of the machine, not of this code.
"""
from __future__ import annotations

import copy
import datetime
import importlib.util
import re

import plotly.graph_objects as go

from . import theme

_probe_cache: tuple[bool, str] | None = None


def available(force: bool = False) -> tuple[bool, str]:
    """Can we render a static image? Returns `(ok, human_readable_reason)`.

    Probes by actually rendering a tiny figure, because importing kaleido
    successfully proves nothing -- it fails at render time when it cannot find
    a browser to drive. The result is cached; pass `force=True` to re-probe.
    """
    global _probe_cache
    if _probe_cache is not None and not force:
        return _probe_cache

    if importlib.util.find_spec("kaleido") is None:
        _probe_cache = (False, "kaleido is not installed. PNG and SVG export need "
                               "it; HTML export works without it.")
        return _probe_cache

    try:
        go.Figure().to_image(format="png", width=200, height=150)
        _probe_cache = (True, "")
    except Exception as exc:
        first_line = str(exc).strip().split("\n")[0][:180]
        _probe_cache = (
            False,
            f"kaleido is installed but could not render ({first_line}). It needs a "
            f"Chrome-family browser; run `plotly_get_chrome` to fetch one. HTML "
            f"export works regardless.",
        )
    return _probe_cache


def slugify(name: str) -> str:
    """A filename-safe stem, so downloads do not collide or need quoting."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "figure"


def filename(name: str, extension: str) -> str:
    """`lilim-<figure>-<timestamp>.<ext>` -- sortable and self-describing."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"lilim-{slugify(name)}-{stamp}.{extension}"


def captioned(fig: go.Figure, caption: str | None,
              width: int = theme.EXPORT_WIDTH) -> go.Figure:
    """A copy of `fig` with the caption attached and room made for it.

    The figure is deep-copied so the on-screen version is never mutated -- the
    app may still be displaying it, and an export must not change what the user
    is looking at.

    The bottom margin is DERIVED from the caption's wrapped line count at the
    export width, not assumed. That is the whole fix: the previous in-figure
    captions used a fixed margin against a height-scaled offset, so they clipped.
    """
    out = go.Figure(copy.deepcopy(fig.to_dict()))

    if not caption:
        out.update_layout(width=width)
        return out

    annotation, margin_bottom = theme.caption_annotation(caption, width)

    existing = list(out.layout.annotations or ())
    out.update_layout(
        width=width,
        annotations=existing + [annotation],
        margin=dict(
            l=out.layout.margin.l if out.layout.margin.l is not None else 64,
            r=out.layout.margin.r if out.layout.margin.r is not None else 24,
            t=out.layout.margin.t if out.layout.margin.t is not None else 56,
            b=margin_bottom,
        ),
    )

    # Grow the canvas by the space the caption just claimed, so the plot itself
    # keeps its original height instead of being squeezed to make room.
    base_height = out.layout.height or theme.H_CHART
    out.update_layout(height=base_height + (margin_bottom - 56))
    return out


def to_png_bytes(fig: go.Figure, caption: str | None = None,
                 scale: int = theme.EXPORT_SCALE) -> bytes:
    """Render to PNG. Raises if kaleido cannot render -- check `available()`."""
    return captioned(fig, caption).to_image(format="png", scale=scale)


def to_svg_bytes(fig: go.Figure, caption: str | None = None) -> bytes:
    """Render to SVG -- vector, so it scales without loss in a typeset document."""
    return captioned(fig, caption).to_image(format="svg")


def to_html_bytes(fig: go.Figure, caption: str | None = None,
                  title: str | None = None) -> bytes:
    """A self-contained interactive HTML page. Never needs an external binary.

    Plotly's JS is inlined rather than pulled from a CDN, so the file works
    offline and will still work years from now, which matters for something
    handed to another research group.
    """
    exported = captioned(fig, caption)
    # The caption is in the figure already; the page wrapper only supplies the
    # surrounding paper colour so it matches the app.
    exported.update_layout(width=None)
    html = exported.to_html(include_plotlyjs=True, full_html=True,
                            config={"displaylogo": False})
    html = html.replace(
        "<body>",
        f'<body style="margin:0;padding:24px;background:{theme.PAPER};">',
        1,
    )
    if title:
        html = html.replace("<head>", f"<head><title>{title} — lilim</title>", 1)
    return html.encode("utf-8")
