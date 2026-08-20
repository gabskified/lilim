"""The visual identity: palette, typography, and a registered Plotly template.

Every figure in `lilim` inherits from the template registered here, so no chart
carries its own styling and the whole app reads as one designed object rather
than a pile of defaults.

DESIGN INTENT
-------------
This is a scientific instrument about trees and heat, so the surface is warm
paper rather than clinical white, the single accent is canopy green, and warm
tones are reserved for meaning -- heat, vulnerability, priority -- never spent
on chrome. Nothing here is a framework default; the accent deliberately
replaces Streamlit's stock red.

Type is a three-way split with distinct jobs: a serif for headings (this is a
research tool, and the serif says so), a humanist sans for interface text, and
a monospace for every number, so digits align in tables and scores stay
comparable at a glance.

COLOUR CHOICES THAT ARE NOT DECORATIVE
--------------------------------------
* Land-use classes get three fixed, semantically-assigned tokens. Prohibited is
  a receding warm grey, Available a living sage, Vulnerable a terracotta that
  reads as "priority" without reading as "error".

* Continuous cooling fields use a perceptually uniform, colourblind-safe ramp.
  This is a DELIBERATE DEPARTURE from the reference implementation, which used
  matplotlib's `RdYlGn_r` for equity and `coolwarm_r` for cooling. Red-yellow-
  green is neither perceptually uniform nor safe for the ~8% of men with
  red-green colour vision deficiency, and it implies a diverging quantity where
  cooling is strictly sequential. The numbers are unchanged; only their
  rendering is.

* Equity weights take three discrete swatches rather than a continuous scale,
  because the implementation applies exactly three levels (1.0 / 1.5 / 2.0). A
  smooth colourbar would imply a continuum that does not exist.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------- palette
INK = "#14201C"           # deep forest black -- primary text
INK_SOFT = "#4A5A52"      # secondary text, axis labels
PAPER = "#F7F5F0"         # warm paper -- page ground
SURFACE = "#FFFFFF"       # cards, plot interiors
RULE = "#DFDAD0"          # hairlines, gridlines, borders

CANOPY = "#2E6B4F"        # the single accent
CANOPY_DEEP = "#1D4633"
CANOPY_LIGHT = "#7FA88C"

TERRACOTTA = "#C4623F"    # heat / vulnerability / priority
AMBER = "#D69A3C"         # mid priority
SLATE = "#5B7C99"         # the contrast arm in comparisons

# Cells where the model delivers exactly zero cooling because the crown-
# competition term has saturated. A deliberately off-scale colour: these are
# real zeros, and against the near-white low end of COOLING_SCALE they were
# indistinguishable from "almost no cooling", which made them read as blank
# discs punched out of the map rather than as a result.
SUPPRESSED = "#B9B2A6"

# Land-use classes. Keys match the integer codes in core.config.
LAND_USE_COLORS = {
    1: "#9A938A",         # Prohibited -- receding warm grey
    3: "#A8C4AC",         # Available  -- living sage
    4: "#C4623F",         # Vulnerable -- priority terracotta
}
LAND_USE_LABELS = {
    1: "Prohibited (built)",
    3: "Available (plantable)",
    4: "Vulnerable (priority)",
}

# Equity weight levels -- three discrete steps, not a continuum.
EQUITY_COLORS = {
    1.0: "#DDE5DC",
    1.5: "#D6A96B",
    2.0: "#B4502F",
}

# Perceptually uniform sequential ramp for cooling intensity: paper -> canopy.
COOLING_SCALE = [
    [0.00, "#F7F5F0"],
    [0.15, "#DCE7DC"],
    [0.35, "#A9CBB4"],
    [0.55, "#6FAB8C"],
    [0.75, "#3D8768"],
    [0.90, "#22624B"],
    [1.00, "#123D2F"],
]

# Categorical sequence for anything with distinct series. Ordered so the first
# few are maximally distinguishable, including in greyscale.
CATEGORICAL = [CANOPY, TERRACOTTA, SLATE, AMBER, "#7B5EA7", "#3F8C8C",
               CANOPY_LIGHT, "#8C6D4F"]

# Sensitivity categories keep stable colours across every chart that shows them.
CATEGORY_COLORS = {
    "Cooling_Model": CANOPY,
    "Weighting": SLATE,
    "Species_Morphology": AMBER,
    "Species_Allometry": TERRACOTTA,
}

# The two scenario arms, fixed so they never swap between figures.
ARM_COLORS = {"WITH": CANOPY, "WITHOUT": SLATE}

# ---------------------------------------------------------------- typography
FONT_SERIF = "'Source Serif 4', 'Iowan Old Style', Georgia, serif"
FONT_SANS = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_MONO = "'JetBrains Mono', 'Cascadia Mono', ui-monospace, monospace"

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2"
    "?family=Inter:wght@400;500;600"
    "&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600"
    "&family=JetBrains+Mono:wght@400;500"
    "&display=swap"
)

TEMPLATE_NAME = "lilim"

# ---------------------------------------------------------------- figure heights
# Named rather than scattered as magic numbers, so figures of the same kind stay
# the same size and a change lands everywhere at once. Charts that must grow with
# their row count (tornado, ranges) compute their own height instead.
H_MAP = 520      # square spatial maps — land use, equity, placement
H_FIELD = 560    # the fine-grid cooling field, which carries more detail
H_CHART = 380    # ordinary single-panel charts
H_SHORT = 320    # compact single-panel charts
H_WIDE = 400     # two-panel (make_subplots) rows

# Width used when rendering a figure to a static image. Chosen so a 3x scale
# lands near 300 dpi at a typical single-column print width.
EXPORT_WIDTH = 1200
EXPORT_SCALE = 3


def build_template() -> go.layout.Template:
    """The Plotly template every figure inherits."""
    axis = dict(
        showgrid=True,
        gridcolor=RULE,
        gridwidth=1,
        zeroline=False,
        linecolor=RULE,
        linewidth=1,
        ticks="outside",
        tickcolor=RULE,
        ticklen=4,
        automargin=True,
        tickfont=dict(family=FONT_MONO, size=11, color=INK_SOFT),
        title=dict(font=dict(family=FONT_SANS, size=12, color=INK_SOFT)),
    )

    return go.layout.Template(
        layout=dict(
            paper_bgcolor=PAPER,
            plot_bgcolor=SURFACE,
            font=dict(family=FONT_SANS, size=12, color=INK),
            title=dict(
                font=dict(family=FONT_SERIF, size=17, color=INK),
                x=0, xanchor="left", pad=dict(b=12),
            ),
            colorway=CATEGORICAL,
            colorscale=dict(sequential=COOLING_SCALE),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor=RULE,
                borderwidth=0,
                font=dict(family=FONT_SANS, size=11, color=INK_SOFT),
            ),
            # Captions live below the figure in the page, not inside it, so the
            # bottom margin only has to hold the axis title. Axes carry
            # automargin so long tick labels push the plot in rather than
            # getting clipped at a narrow width.
            margin=dict(l=64, r=24, t=56, b=56),
            autosize=True,
            hoverlabel=dict(
                bgcolor=SURFACE,
                bordercolor=RULE,
                font=dict(family=FONT_MONO, size=11, color=INK),
            ),
            separators=".,",
        )
    )


def register() -> str:
    """Register the template and make it the default. Idempotent."""
    pio.templates[TEMPLATE_NAME] = build_template()
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


CAPTION_FONT_SIZE = 12
CAPTION_LINE_HEIGHT = 17
CAPTION_PAD = 18


def wrap_caption(text: str, width_px: int, font_size: int = CAPTION_FONT_SIZE) -> list[str]:
    """Break a caption into lines that fit `width_px`, for export rendering.

    Plotly annotations do not wrap on their own, so the wrapping has to happen
    here. The character-width estimate (0.52 em for this sans at these sizes) is
    approximate by nature, which is exactly why callers should size the margin
    from the returned line count rather than guessing an offset.
    """
    approx_char_px = font_size * 0.52
    max_chars = max(20, int(width_px / approx_char_px))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def caption_annotation(text: str, width_px: int) -> tuple[dict, int]:
    """A caption annotation plus the bottom margin needed to hold it.

    Returns `(annotation, margin_bottom_px)`. The margin is DERIVED from the
    wrapped line count rather than assumed — the previous version of this
    helper hard-coded a paper-coordinate offset against a fixed 52px margin,
    which clipped or collided on essentially every figure, because paper offsets
    scale with each figure's own height. Anchoring in pixels below the plot area
    and sizing the margin to match is what makes that impossible.
    """
    lines = wrap_caption(text, width_px - 88)
    text_height = len(lines) * CAPTION_LINE_HEIGHT
    margin_bottom = 56 + CAPTION_PAD + text_height

    ann = dict(
        text="<br>".join(lines),
        showarrow=False,
        xref="paper", yref="paper",
        x=0, xanchor="left",
        y=0, yanchor="top",
        # Shift down in PIXELS from the bottom of the plot area, so the offset
        # does not scale with figure height the way a paper offset does.
        yshift=-(56 + CAPTION_PAD),
        align="left",
        font=dict(family=FONT_SANS, size=CAPTION_FONT_SIZE, color=INK_SOFT),
    )
    return ann, margin_bottom


def css() -> str:
    """The stylesheet injected into Streamlit, so the app matches the figures.

    Streamlit's own theme is set in `.streamlit/config.toml`; this covers what
    config.toml cannot reach -- the serif headings, monospace metrics, and the
    flatter, quieter surfaces.
    """
    return f"""
    @import url('{GOOGLE_FONTS_URL}');

    html, body, [class*="css"] {{
        font-family: {FONT_SANS};
        color: {INK};
    }}
    h1, h2, h3, h4 {{
        font-family: {FONT_SERIF} !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
        color: {INK} !important;
    }}
    h1 {{ font-size: 1.9rem !important; }}
    h2 {{ font-size: 1.4rem !important; margin-top: 1.6rem !important; }}
    h3 {{ font-size: 1.1rem !important; color: {INK_SOFT} !important; }}

    code, pre, .stMetric [data-testid="stMetricValue"] {{
        font-family: {FONT_MONO} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.5rem !important;
        color: {CANOPY_DEEP} !important;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: {FONT_SANS} !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {INK_SOFT} !important;
    }}

    section[data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {RULE};
    }}
    section[data-testid="stSidebar"] h2 {{
        font-size: 1.05rem !important;
        margin-top: 1.1rem !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 1.6rem;
        border-bottom: 1px solid {RULE};
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: {FONT_SANS};
        font-size: 0.9rem;
        letter-spacing: 0.01em;
        padding: 0.4rem 0;
    }}

    .stButton > button {{
        font-family: {FONT_SANS};
        border-radius: 3px;
        border: 1px solid {CANOPY};
        background: {CANOPY};
        color: {PAPER};
        font-weight: 500;
        letter-spacing: 0.01em;
    }}
    .stButton > button:hover {{
        background: {CANOPY_DEEP};
        border-color: {CANOPY_DEEP};
        color: {PAPER};
    }}

    hr {{ border-color: {RULE}; }}

    /* A quiet note block, for caveats that must travel with a number. */
    .lilim-note {{
        font-family: {FONT_SANS};
        font-size: 0.82rem;
        line-height: 1.5;
        color: {INK_SOFT};
        border-left: 2px solid {RULE};
        padding: 0.1rem 0 0.1rem 0.8rem;
        margin: 0.5rem 0 1rem 0;
    }}
    .lilim-note strong {{ color: {INK}; font-weight: 600; }}

    .lilim-rule {{
        height: 1px; background: {RULE};
        margin: 1.4rem 0 1.1rem 0; border: 0;
    }}

    /* Figure captions. These live in the page, NOT inside the Plotly figure:
       an in-figure caption is positioned in paper coordinates that scale with
       the figure's own height, so it clips or collides as the viewport moves.
       As flowing HTML it simply reflows and can never be cropped. */
    .lilim-caption {{
        font-family: {FONT_SANS};
        font-size: 0.78rem;
        line-height: 1.5;
        color: {INK_SOFT};
        margin: -0.4rem 0 0.4rem 0;
        padding: 0 0.2rem 0 0.2rem;
        max-width: 68ch;
    }}

    /* ---------------------------------------------------------------
       Responsive stacking.

       Streamlit columns are flex children that shrink rather than wrap, so a
       side-by-side chart pair squeezes into unreadability on a narrow window
       instead of stacking. These breakpoints make them wrap. Selectors target
       Streamlit's internal test ids, which are verified against the version
       pinned in requirements.txt — recheck them after a Streamlit upgrade.
       --------------------------------------------------------------- */
    @media (max-width: 1100px) {{
        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
        }}
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}
    }}

    /* Metric rows are four across at full width. Let them go two-up before
       they collapse entirely, so the numbers stay readable in between. */
    @media (min-width: 761px) and (max-width: 1100px) {{
        [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])
            > [data-testid="stColumn"] {{
            flex: 1 1 46% !important;
            min-width: 46% !important;
        }}
    }}

    /* Keep the plot area from being squeezed below a usable width. */
    .stPlotlyChart {{ min-width: 0; }}
    """
