# Phase 0 — EchoNet-Dynamic Baseline Reproduction

## Experimental setup

I reproduced the EchoNet-Dynamic pipeline using the official data splits. After the loader's tracing checks, 7,460 training, 1,288 validation, and 1,276 test videos were available.

I trained two models separately. DeepLabV3-ResNet50 learned to segment the left ventricle from manually traced frames. R(2+1)D-18 learned to predict left ventricular ejection fraction (EF) from 32-frame video clips sampled at a stride of 2.

## Test results

| Evaluation | Metric | Result | Videos |
| --- | --- | ---: | ---: |
| Left ventricular segmentation | Overall Dice | 0.9230 | 1,276 |
| End-diastolic segmentation | Dice | 0.9327 | 1,276 |
| End-systolic segmentation | Dice | 0.9078 | 1,276 |
| Beat-averaged EF prediction | MAE | 4.1052 percentage points | 1,274 |
| Beat-averaged EF prediction | RMSE | 5.4342 percentage points | 1,274 |
| Beat-averaged EF prediction | R² (linear fit) | 0.8029 | 1,274 |

Two test videos had no detected contraction that could be matched to an available clip prediction, so they were excluded from the beat-averaged EF metrics.

## Single-video walkthrough

The segmentation model produced a left ventricular area curve. Four detected contractions at frames 20, 85, 150, and 211 mapped to clip starts at frames 0, 53, 118, and 179. The four clip predictions averaged to 58.7332% EF. The reference EF was 55.9518%, giving an absolute error of 2.7814 percentage points.
