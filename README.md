# lilim

*Tagalog: **lilim** — the shade a tree casts.*

An interactive, modular reference implementation of **SECPI**, the Synergistic
and Equitable Cooling Performance Index: a framework for optimising urban tree
placement across six Philippine tree functional types, weighting cooling
benefit by the vulnerability of the people who receive it.

This directory is the codebase handed to whoever picks this research up next.
It exists because the version that produced the published results was a single
3,700-line file with no interface, and the harnesses that regenerated those
results dropped visualization entirely — across seven full regeneration runs,
not one chart was ever drawn. Every number was correct and no one could see
anything.

---

## Quick start

**Windows (PowerShell)** — note that PowerShell does not accept `&&` as a
statement separator, so these are separate lines, not a chain:

```powershell
cd lilim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python lilim.py
```

**macOS / Linux:**

```bash
cd lilim
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python lilim.py
```

`python lilim.py` opens the workbench in a **native application window** — its
own title bar, no browser, no URL to copy. It remembers its size and position
between launches.

If you would rather have it in a browser tab, or the desktop window will not
open on your platform, the underlying app runs standalone:

```
streamlit run workbench.py
```

Then verify it against the reference implementation before trusting a number:

```
python parity/check_parity.py
```

That should print `0 failures`. If it does not, this codebase is wrong and the
reference implementation is right — see [Parity](#parity-the-load-bearing-claim).

---

## What you can do with it

| Tab | What it shows |
|---|---|
| **Grid** | The cellular-automata land-use grid and the equity-weight map, as charts |
| **Species & cooling** | Radial decay curves per species and the cooling-potential breakdown |
| **Optimize** | An optimizer run with the convergence chart **filling in live**, then the placement, cooling field, class distribution, and zonal efficiency |
| **Scenarios** | The tree-count sweep across both equity arms, with per-restart spread |
| **Sensitivity** | The 40-factor one-at-a-time sweep as tornado, category, and range charts |
| **About** | What this is, what it is not, and where the real provenance record lives |

Every result the app can compute has a chart. Nothing is tables-only.

### Saving figures

Every chart has a **Save figure** control beneath it, offering PNG at 300 dpi,
SVG (vector, for typesetting), or a self-contained interactive HTML file.

**The caption is written into the saved file.** On screen, captions are page
text below each chart — that is the only way they stay aligned at any window
size. But a figure that gets separated from its caveat is how a qualified
result quietly becomes an unqualified one, so the export path re-attaches the
caption to the figure itself and sizes the margin to fit it.

PNG and SVG need `kaleido`, which drives a Chrome-family browser to rasterise.
If it cannot find one, run `plotly_get_chrome`; the app detects this and falls
back to HTML export, which needs no external binary and always works. Note that
HTML exports inline the whole Plotly library (~5 MB each) so they stay readable
offline and years from now — a deliberate trade for an archival artifact.

Two analyses are in `core/` but deliberately not wired into the app, because
neither can run at interactive speed. **The species-palette sweep** — all 63
subsets of the six-species palette, 630 seeded restarts — is available as
`core.scenarios.run_subset_sweep()`, with `viz.figures.palette_size_effect()`
to chart it. **Multi-grid Morris sensitivity and morphological robustness
validation** are not here at all; they live in `tools/` and run in the tens of
thousands of evaluations.

---

## How this relates to the manuscript and to `legacy/AuditedCode_1.py`

Three things exist and they have different jobs:

**`legacy/AuditedCode_1.py`** is the **audited reference of record**. It is what
the manuscript's numbers were produced by, it is what carries the audit trail,
and it stays authoritative. Where it and `lilim` disagree, assume `lilim` is
wrong until proven otherwise — with exactly one documented exception, below.

**`tools/`** holds the batch harnesses that regenerated the reported results:
`tools/secpi_results/` for the main sweeps and the paired statistical tests,
`tools/morris/` for the multi-grid Morris sensitivity analysis. Those are jobs,
not tools you sit in front of. `lilim`'s sweep and seeding logic is adapted
from them rather than re-derived.

**`lilim/`** — this directory — is the same science, reorganised so it can be
read, run, and seen. It is verified equal to the reference implementation to
the last bit. It is *not* a replacement for the audit trail, and it does not
supersede anything.

---

## Layout

```
lilim/
├── lilim.py                desktop launcher — native window, no browser
├── workbench.py            the Streamlit app — presentation only, no science
├── core/
│   ├── config.py           every production constant, one source of truth
│   ├── species.py          the six tree functional types and their parameters
│   ├── grid.py             cellular-automata land-use generation, equity weights
│   ├── cooling.py          Gaussian radial decay with crown-competition damping
│   ├── aco.py              the Ant Colony System optimizer
│   ├── secpi.py            cutoff calibration, the index, normalization, metrics
│   ├── scenarios.py        tree-count and species-palette sweeps, both arms
│   └── sensitivity.py      the 40-factor one-at-a-time sweep
├── viz/
│   ├── theme.py            palette, typography, the Plotly template
│   ├── figures.py          one function per chart, plus their captions
│   └── export.py           saving figures with their caption attached
├── parity/
│   ├── check_parity.py     verification against the reference implementation
│   └── PARITY_REPORT.md    the committed result, with numbers
└── requirements.txt
```

Two rules hold across the whole tree. `core/` never draws anything and never
imports `viz/`. `viz/` and `workbench.py` never compute anything scientific —
they render what `core/` produced.

---

## The science, briefly

The domain is a **synthetic, non-georeferenced** 100 × 100 m block: a 10 × 10
grid of 10 m coarse cells, which are the planting units, over a 1 m fine grid,
which is where cooling is evaluated.

A cellular automaton generates land use in three phases — Almeida-style
stochastic growth for realistic clustering, deterministic calibration to the
target density, then vulnerable-zone carving by breadth-first expansion. Cells
end up **Prohibited** (built, unplantable), **Available** (plantable), or
**Vulnerable** (plantable neighbours get priority).

Each tree cools its surroundings with Gaussian radial decay scaled by its own
crown diameter, `exp(-λ(d/C_D)²)`, damped where crowns overlap. An Ant Colony
System searches over (cell, species) pairs to maximise SECPI, which scores how
the domain's cooling distributes across four benefit classes, weighted by class
rank and by the equity weight of the cells in each class.

Cutoffs for those four classes are **fixed study-wide**, calibrated once by
Monte Carlo pooling and reused everywhere. This is load-bearing: under the
earlier self-normalizing scheme, every scenario took its own quartiles, so
every placement landed at roughly 25/25/25/25 regardless of how much cooling it
actually delivered, and comparing scenarios was meaningless.

---

## Parity: the load-bearing claim

A rewrite that silently diverges from the audited implementation is worse than
no rewrite. `parity/check_parity.py` therefore runs identical scenarios through
both codebases and compares by **exact equality** — `np.array_equal` and `==`,
never a tolerance.

```bash
python parity/check_parity.py                  # stages 0-3 and 5
python parity/check_parity.py --with-oat       # adds the sensitivity sweep
python parity/check_parity.py --report OUT.md  # write a markdown report
```

| Stage | What it checks | Result |
|---|---|---|
| 0 | Grid generation at seed 42 — land-use array, fine grid, plantable and vulnerable coordinates, all 10,000 equity weights | identical |
| 1 | Study-wide reference cutoffs (Q1, Q2, Q3) | identical |
| 2 | One seeded optimizer restart — score, placement, species, the full 10,000-point cooling field, both convergence traces | identical |
| 3 | The full tree-count sweep — 60 seeded restarts across both arms | identical to live reference **and** to the committed regeneration output |
| 4 | The 40-factor sensitivity sweep, plus proof shared species state was restored | identical |
| 5 | The coarse equity-weight map | **diverges by design** — see below |

Stage 3 is the strongest check available: those 60 values are compared not only
against a live run of the reference implementation but against
`results/run_20260820_003124_optionB_results_regeneration/data/headline.json`,
which is where the reported numbers actually come from. So this codebase is
tied directly to the manuscript's own record, not merely to code that agrees
with itself.

See [`parity/PARITY_REPORT.md`](parity/PARITY_REPORT.md) for the committed run.

### Reproducibility is fragile in a specific way

Determinism here rests entirely on **global numpy random state and the order
draws are consumed in** — `np.random.seed` / `shuffle` / `random` / `choice` /
`uniform` / `randint`, in a fixed sequence. There is no `Generator` object and
no seed threading.

That means modernising the RNG, vectorising the cellular automaton's per-cell
Bernoulli trial, or reordering a loop would each silently change every
downstream number while looking like a clean refactor. Those places are marked
with warnings in the source. **Run the parity check after touching anything in
`core/`.**

---

## The one deliberate difference

`get_coarse_cell_weights` — the function behind the coarse equity-weight map —
indexes the fine-resolution weight array **row-major** in the reference
implementation, while that array is ordered **x-major** over `fine_grid_points`.
On the square study grid the two conventions differ by exactly a transpose, so
the map the reference implementation draws is its own mirror image about the
diagonal. Measured: maximum absolute difference 1.000000, and the two arrays
are exact transposes of each other.

`lilim` uses the correct index, and Stage 5 validates it against a ground truth
computed from coordinates alone — so the fix is verified against something that
cannot share its failure mode, not merely self-consistent.

**Nothing else is affected.** That function's only consumer is the equity-weight
figure. Every score, placement, cooling field, and sensitivity result in this
project comes from `_calculate_vulnerability_weights` and
`calculate_total_cooling`, both of which are coordinate-based and both of which
are confirmed identical between the two codebases.

Correcting it in `legacy/AuditedCode_1.py` is a semantic change to an audited
codebase and needs author authorisation, which is still pending there. This
directory is not that audited codebase, so the fix is applied here directly.
**Expect the two to disagree on that one figure, and expect `lilim` to be the
correct one.**

---

## Where the real record lives

This code is the *what*. The *why* is in the project's own documents, and the
next group should start there rather than reverse-engineering intent from
source. Nearly every non-obvious choice in `core/` — the fixed study-wide
normalization, the tree-count sweep range, the species-height assumptions, the
state-isolation guard in the sensitivity sweep, the corrected cellular-automaton
recursion — traces to a specific, dated, authorised entry:

- **[`docs/DECISIONS.md`](../docs/DECISIONS.md)** — every semantic change made
  to the reference implementation: what changed, why, who authorised it, and
  which commit applied it. Read this first.
- **[`docs/PROJECT_LOG.md`](../docs/PROJECT_LOG.md)** — the append-only
  evidentiary trail. Where a decision's reasoning is not fully captured in the
  decision itself, it is here.
- **[`CLAUDE.md`](../CLAUDE.md)** — the project's scope constraints and its
  standing rules, including the two that matter most: never invent data, and
  never declare something verified by reading it rather than running it.

---

## Known open items

These are open questions in the science, not bugs in this code. A tool that
hides them is worse than one that does not exist.

- **The coarse equity-weight map** here is corrected; the audited reference
  implementation's version is still its own transpose, pending author sign-off
  on the fix. The two will disagree on that one figure.
- **Three of the six species' heights are author estimates**, not measurements.
  They sit near the top of the observed field-data range, which makes the
  cooling potentials derived from them optimistic if the estimates are high.
- **Adopted leaf-area values are not derived from the allometric constants
  published beside them.** The allometric pipeline (`DBH = (h/h₀)^(1/h₁)`,
  `LAI = l₀·DBH^l₁`) is implemented and returns values roughly two orders of
  magnitude smaller than the adopted figures. The analysis uses the adopted
  column; the allometric path is exercised only by the sensitivity sweep, and
  there only through relative change.
- **The sensitivity-analysis numbers in the originally published manuscript are
  not reproducible from this code.** The current values come from re-running
  the analysis, not from reconciling with the original, and no forensic search
  for the original source was undertaken.
- **The 0–5 SECPI scale does not place the no-intervention baseline at zero.**
  It is a min–max transform against derived theoretical bounds, and a raw score
  of 0 maps to 0.588. The transform is strictly increasing, so it never changes
  which placement wins, but the reported scale has no natural origin.
- **Equity weighting runs as three discrete levels** (1.0 / 1.5 / 2.0) by
  distance from the nearest vulnerable cell. This has not been reconciled
  against the manuscript table describing a continuous 0.5–2.0 multiplier range.
- **The vulnerable-zone construction is a breadth-first expansion to a target
  count**, not the 30 m Chebyshev buffer the Methods section describes. The
  literal buffer is geometrically incompatible with a 5–10% target at this grid
  size; the implementation has always done the former.
- **The domain is synthetic and non-georeferenced.** There is no field site, no
  remote-sensing validation, and no raster of any real city. Any reading of the
  outputs as a literal map is wrong.
- **The optimizer has no diversity-enforcement mechanism.** It is Ant Colony
  System with a pseudo-random-proportional action rule. A solution that
  converges on a single species reflects the objective and the heuristic, not a
  constraint — and in practice it frequently does converge that way.
- **Multi-grid Morris sensitivity and morphological robustness validation are
  not in this app.** They are batch analyses in the tens of thousands of
  evaluations and live in `tools/`.

---

## Design notes

The interface deliberately does not use Streamlit's default theme. The accent
is canopy green, the ground is warm paper, and warm tones are reserved for
meaning — heat, vulnerability, priority — rather than spent on chrome. Type
splits three ways by job: serif for headings, humanist sans for interface,
monospace for every number so digits align and scores stay comparable.

Charts depart from the reference implementation's colours in one respect worth
stating: continuous cooling fields use a perceptually uniform, colourblind-safe
ramp instead of `RdYlGn_r`, which is neither, and which implies a diverging
quantity where cooling is strictly sequential. Equity weights get three
discrete swatches rather than a smooth colourbar, because the implementation
applies exactly three levels and a gradient would imply a continuum that does
not exist. **Only the rendering changed; no number did.**

---

## Citation and authorship

The underlying research is by Lacuanan, Valenzuela, De Leon, Villadolid,
Valdes, and Suarez (2025), Caloocan City Science High School. Cite the
manuscript, not this directory. If you extend this code, the parity check is
the contract — keep it passing, and add stages when you add science.
