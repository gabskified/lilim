# Parity report — `lilim/core` against `legacy/AuditedCode_1.py`

Generated 2026-08-20 19:39 · numpy 2.5.2 · grid seed 42 · 653.0s

Comparison is by exact equality (`np.array_equal`, `==`), not tolerance,
except for the single documented divergence in Stage 5.

| Stage | Checks | Failures | Seconds |
|---|---|---|---|
| 0 — grid generation | 6 | 0 | 0.3 |
| 1 — reference cutoffs | 1 | 0 | 0.4 |
| 2 — single seeded restart | 9 | 0 | 7.8 |
| 3 — k sweep | 4 | 0 | 217.7 |
| 4 — OAT sensitivity | 4 | 0 | 425.8 |
| 5 — equity-weight map | 3 | 0 | 0.0 |

## Every check

| Stage | Check | Result | Detail |
|---|---|---|---|
| 0 | grid.coarse_grid | pass | 100 values identical |
| 0 | grid.fine_grid | pass | 10000 values identical |
| 0 | grid.plantable_coords | pass | 62 values identical |
| 0 | grid.vulnerable_coords | pass | 1600 values identical |
| 0 | grid.vulnerability_weights | pass | 10000 values identical |
| 0 | land-use composition in target bands | pass | P=61 A=31 V=8 plantable=31 |
| 1 | reference cutoffs (Q1, Q2, Q3) | pass | (0.000000000011, 0.000052713867, 0.020532203314) |
| 2 | best raw SECPI | pass | np.float64(2.977910453709381) (seed 4000500) |
| 2 | best-solution coordinates | pass | 10 values identical |
| 2 | best-solution species | pass | Narra + Banaba + Narra + Narra + Akleng-parang |
| 2 | best cooling field | pass | 10000 values identical |
| 2 | history_best per iteration | pass | 40 values identical |
| 2 | history_avg per iteration | pass | 40 values identical |
| 2 | iteration callback is inert (score unchanged) | pass | 40 callbacks fired |
| 2 | iteration callback is inert (history unchanged) | pass |  |
| 2 | callback saw every iteration | pass | 40 of 40 |
| 3 | WITHOUT-arm equity weights | pass | 10000 values identical |
| 3 | WITHOUT-arm coarse grid | pass | 100 values identical |
| 3 | 60 restarts match live legacy exactly | pass | all 60 identical |
| 3 | 60 restarts match the committed regeneration output | pass | 60 values from headline.json |
| 4 | factor count | pass | 40 factors |
| 4 | baseline SECPI | pass | 3.292729882429785 |
| 4 | all 40 sensitivity indices match | pass | identical |
| 4 | shared species state restored after sweep | pass | SPECIES_DATA unchanged |
| 5 | lilim map matches coordinate-derived ground truth | pass | 100 values identical |
| 5 | legacy map is the transpose of the corrected one | diverges by design | max abs difference 1.000000 |
| 5 | divergence is confined to this function | pass | fine-resolution equity weights identical in both |

**27 checks, 0 failures.**
