# Phase 2 evaluation run record

Three frozen EF systems were evaluated on the same 644 videos from the official validation split. Each video was evaluated under clean, gamma, contrast and blur conditions using fixed frame indices. Training updates performed in Phase 2: 0.

| Slurm job | Operation | Recorded outcome |
|---|---|---|
| 541 | Prepare normalization and frame indices | Indices saved for all 644 videos. |
| 542 | Smoke test on 16 videos with all three systems | Completed with 0 failures. |
| 543 | Full evaluation of R(2+1)D-18 | 644/644 videos, four conditions, 0 failures. |
| 544 | Full evaluation of X3D-M S-Sup and PanEcho | 644/644 videos per system, four conditions, 0 failures. |

Both full jobs logged the completion marker. The [job 543](logs/job543_progress.txt) and [job 544](logs/job544_progress.txt) files contain progress, repeatability and completion lines selected from the corresponding Slurm output.

## Checks and results

- All parameters stayed frozen, and frame indices matched across conditions within each system.
- Repeat clean inference on the first 16 videos produced identical outputs for all three systems. The official test set was not used.
- All [104 independent checks](summary/independent_check.csv) and the recorded final audit passed.
- [Coverage](summary/coverage.csv), [MAE](summary/accuracy.csv), [prediction drift and Bland–Altman limits](summary/drift_bland_altman.csv).
- [Phase 2 report and figures](../../reports/phase2/PHASE2_REPORT.md) and [original inference and analysis workflow](../../scripts/phase2/README_RUN.md).
