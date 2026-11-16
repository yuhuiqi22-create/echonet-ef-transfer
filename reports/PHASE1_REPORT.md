# EchoNet-Dynamic Phase 1 Report

**Teacher:** PanEcho (EF output only, frozen) · **Student:** X3D-M (single EF regression output)
**Data:** EchoNet-Dynamic, official TRAIN for training; official VAL split into 644 model-selection and 644 pilot-evaluation videos; official TEST not used.
**Status:** Tasks 2–4, 4 student conditions × 3 seeds. No stop condition was triggered.

## Summary

We selected X3D-M trained with standard data augmentation (S-Aug) as the compact EF model for Phase 1 and kept PanEcho as a reference. S-Aug had the lowest mean MAE among the student models on clean inputs and in all three stress tests. The current evidence is insufficient to justify developing a new method in Phase 2.

---
# Task 1. Concepts

| Topic | Explanation |
|---|---|
| Knowledge distillation | A trained teacher gives the student an extra learning signal. In S-KD, PanEcho and X3D-M see the same 16-frame clip, and X3D-M learns from both the EF label and PanEcho's prediction. |
| Regression KD loss | The loss combines two squared-error terms, one for the EF label and one for the teacher's prediction. I weighted them equally (`α = 0.5`). The teacher's prediction changes the EF value the student is encouraged to output. |
| PanEcho | PanEcho is a pretrained model for several echocardiography tasks, including EF prediction. I checked that its predictions agreed on repeated runs of the same clip in Task 2 and used its frozen weights to generate teacher targets in Task 3. |
| X3D-M | X3D-M is a compact video model whose backbone extracts spatial and motion features. I replaced its classification head with a single-output EF regression head and fine-tuned both the backbone and the new head. The S-Sup run with seed `1001` served as the first baseline for Task 3. |
| Training seed | A seed controls random choices during training. I used `1001`, `1002`, and `1003` to vary head initialization, sample order, and clip sampling, then checked whether the results were consistent across runs. |
| Mean ± SD | The mean summarizes performance across runs, and the standard deviation (SD) measures how much it varies. S-Sup's MAE of **5.932 ± 0.179 pp** is the mean and SD across three training seeds, in percentage points. |
| Bootstrap CI | A paired bootstrap confidence interval measures uncertainty from the evaluation sample by resampling videos with replacement and keeping both models' errors together. Across the 644 pilot videos, S-Aug-KD minus S-Aug had a mean MAE difference of **+0.167 pp** with a **95% CI of [+0.007, +0.322] pp**. Seed SD measures variation across training runs. |
| Augmentation vs. stress test | Augmentation changes inputs during training, while stress testing measures how trained models respond to fixed input changes. I used random brightness, contrast, and gamma adjustments for training and fixed gamma changes, blur, and depth attenuation for evaluation. |
| Efficiency metrics | Parameter count measures model size, latency measures prediction time, and peak GPU memory measures the highest allocation during inference. On the same GPU and at the same precision, median single-clip latency was **10.46 ms** for X3D-M and **25.49 ms** for PanEcho. |
| Data separation | Each split has a specific purpose. I used `TRAIN` to fit the student, `model_selection` to choose checkpoints, and `pilot_eval` to compare models. The last two each had 644 videos from the official validation split.. |
# Task 2. Set Up the Teacher and Train the Supervised Compact Baseline
## A. Prepare the PanEcho Teacher

I loaded the public PanEcho v1.0 checkpoint (`panecho.pt`), froze its weights, and verified that repeated inference on the same clip gave matching EF predictions. The PanEcho paper lists EchoNet-Dynamic as an external validation dataset, and the repository documents fine-tuning on it as a separate experiment. [PanEcho paper](https://doi.org/10.1001/jama.2025.8731) · [Official repository](https://github.com/CarDS-Yale/PanEcho)

## B. Prepare the X3D-M Student

Both models received the same 16 frames in chronological order at 224 × 224, with their own normalization. All student conditions used the same training videos and clip-sampling rule.

## C. Create the Development Split

I split the official validation set once with seed `2026` into 644 videos for checkpoint selection (`model_selection`) and 644 for comparing models (`pilot_eval`). I used the official `TRAIN` split for training.

## D. Train the Supervised Baseline

I trained S-Sup for 10 epochs with seed `1001`, using one random 16-frame clip per video per epoch and the reference EF as the target. I updated the backbone and regression head using MSE and AdamW. Epoch 10 had the lowest `model_selection` MAE and served as the supervised baseline for Task 3.

### Deliverable A: Model and Input Check

| Item | PanEcho Teacher | X3D-M Student |
|---|---|---|
| Checkpoint / initialization | CarDS-Yale/PanEcho release v1.0 `panecho.pt` via `torch.hub`, `tasks=["EF"]`, eval mode, all parameters frozen | PyTorchVideo X3D-M, Kinetics-400 pretrained; classification head replaced by `Linear(·, 1)` |
| EchoNet-Dynamic used during pretraining or fine-tuning? | The repository describes EchoNet-Dynamic only in separate fine-tuning experiments with their own outputs, and gives no indication that the released checkpoint was trained on it. This could not be verified independently. Not fine-tuned in Phase 1. | Trained on EchoNet-Dynamic TRAIN only (7,465 videos) |
| Input frames | 16 consecutive source frames, identical for both models | Same 16 frames |
| Input image size | 224 × 224 (each 112 × 112 frame resized to 256, centre crop 224) | Same image content |
| Frame order | Chronological | Chronological |
| Normalization | ImageNet mean/std, OpenCV (BGR) channel order as in the PanEcho loader | /255, mean 0.45, std 0.225, RGB |
| EF output scale | EF in % | Trained on EF/100; ×100 for evaluation (%) |
| Parameter count | 42,019,681 (EF-only model) | 2,976,723 |

The PanEcho checkpoint loaded, ran in eval mode, and gave stable predictions for identical inputs: its 644 per-video pilot predictions agreed with job 429 to 6 decimal places in all 11 later clean evaluation jobs.

### Deliverable B: Initial Clean Performance

One prediction per video: 4 evenly spaced 16-frame clips, with clip predictions averaged per video.

| Model | Training Seed | Pilot-Evaluation Videos | MAE | RMSE |
|---|---:|---:|---:|---:|
| PanEcho teacher | Fixed checkpoint | 644 | 5.782 | 7.868 |
| S-Sup | 1001 | 644 | 5.920 | 7.771 |

### Deliverable C: Initial Efficiency Check

NVIDIA GB10, FP32 (TF32 disabled), batch size 1, 30 warm-up and 100 timed runs. **Timing scope:** a synchronized forward pass on a preallocated synthetic GPU clip only, with no decoding or preprocessing. Peak memory is PyTorch-allocated CUDA memory, including the model and input.

| Model | Input Protocol | Parameters | Single-Clip Latency (median / P95) | Peak Memory |
|---|---|---:|---:|---:|
| PanEcho teacher | 16 frames, 224 × 224 | 42,019,681 | 25.49 / 26.19 ms | 364.2 MiB |
| X3D-M student | 16 frames, 224 × 224 | 2,976,723 | 10.46 / 10.62 ms | 136.9 MiB |
| Phase 0 R(2+1)D reference | 32 frames, period 2, 112 × 112 (one clip) | 31,300,638 | 37.15 / 38.06 ms | 320.2 MiB |

X3D-M has a clear cost advantage (Task 2 continuation condition met).

# Task 3. Standard KD and augmentation

All students use the same X3D-M architecture, recipe and selection rule:
- AdamW, learning rate 1e-4, weight decay 1e-4, batch 4, 10 epochs.
- One random 16-frame clip per TRAIN video per epoch.
- The checkpoint is the epoch with the lowest model_selection MAE.
Clip positions and shuffling are identical across conditions for a given seed. KD uses α = 0.5 with PanEcho's EF prediction for the same clip.

## A. Standard Distillation Baseline

With seed 1001, S-KD had an MAE 0.20 pp lower than S-Sup (paired CI [−0.43, +0.02] pp). The interval included zero, so we ran seeds 1002 and 1003 before moving to Task 3B.

### Deliverable A: Supervised vs. KD

Pilot-evaluation MAE/RMSE (pp), 644 videos, each seed model evaluated separately (no ensembling).

| Model | Seed 1001 MAE | Seed 1002 MAE | Seed 1003 MAE | MAE Mean ± SD | RMSE Mean ± SD |
|---|---:|---:|---:|---:|---:|
| S-Sup | 5.920 | 6.118 | 5.760 | 5.932 ± 0.179 | 7.908 ± 0.242 |
| S-KD | 5.717 | 5.949 | 5.880 | 5.849 ± 0.119 | 7.763 ± 0.172 |

Paired S-KD − S-Sup per seed, with 95% video bootstrap CIs:
- 1001: −0.203 [−0.426, +0.019]
- 1002: −0.169 [−0.421, +0.084]
- 1003: +0.120 [−0.107, +0.350]

## B. Add a Standard Augmentation Control

**Augmentation (S-Aug, S-Aug-KD):**
- Changes (appearance only): brightness ×U(0.85, 1.15), contrast ×U(0.85, 1.15), gamma log-uniform in [0.8, 1.25].
- One draw per clip, applied identically to all 16 frames. No temporal, geometric, crop or frame-count change.
- Both augmented conditions receive identical augmented clips, and in S-Aug-KD the teacher and student see the same augmented tensor.

### Deliverable B: Full 2 × 2 Training Comparison

| Model | Extra Image Augmentation | PanEcho KD | MAE Mean ± SD | RMSE Mean ± SD |
|---|---|---|---:|---:|
| S-Sup | No | No | 5.932 ± 0.179 | 7.908 ± 0.242 |
| S-KD | No | Yes | 5.849 ± 0.119 | 7.763 ± 0.172 |
| S-Aug | Yes | No | 5.759 ± 0.078 | 7.775 ± 0.121 |
| S-Aug-KD | Yes | Yes | 5.926 ± 0.105 | 7.914 ± 0.251 |


### Deliverable C: Training Stability

| Model | Seeds Completed | Best-Checkpoint Epochs | Any Training Failure or Instability? |
|---|---|---|---|
| S-Sup | 1001, 1002, 1003 | 10, 5, 9 | No. All losses and metrics finite; model_selection MAE fluctuates between epochs (range 5.2–11.9 pp across S-Sup runs), so the epoch choice matters. |
| S-KD | 1001, 1002, 1003 | 10, 8, 9 | No. Seed 1003's first job (442) failed before training because of an external HTTP 504 (see Notes) and was rerun unchanged as job 446. |
| S-Aug | 1001, 1002, 1003 | 10, 9, 9 | No |
| S-Aug-KD | 1001, 1002, 1003 | 10, 8, 9 | No |

Every run was checked for the following:
- 10 finite epochs.
- The saved best epoch is the epoch with the lowest model_selection MAE.
- The checkpoint loads strictly into X3D-M with finite weights.
- The recorded seed and settings match the recipe.

# Task 4. Input-change robustness and final efficiency

**Robustness test setup.** We used four TRAIN videos to choose three perturbations and check that cardiac structures remained visible: gamma darkening (γ = 1.5), Gaussian blur (σ = 1.5 px), and linear depth attenuation from the sector apex (strength = 0.45). We fixed these settings before pilot evaluation.
The same clips from 644 pilot-evaluation videos under clean and perturbed conditions. After resizing and cropping, each perturbation used the same settings across all 16 frames of a clip, with frame order and clip positions unchanged.

## Deliverable A: Stress-Test Results (MAE, pp)

| Model | Clean MAE | Gamma MAE | Blur MAE | Depth-Attenuation MAE |
|---|---:|---:|---:|---:|
| PanEcho teacher | 5.782 | 5.699 | 8.613 | 5.887 |
| S-Sup, mean ± SD | 5.932 ± 0.179 | 6.368 ± 0.532 | 7.392 ± 1.189 | 6.044 ± 0.214 |
| S-KD, mean ± SD | 5.849 ± 0.119 | 6.136 ± 0.134 | 7.077 ± 1.075 | 5.920 ± 0.190 |
| S-Aug, mean ± SD | 5.759 ± 0.078 | 5.899 ± 0.126 | 6.339 ± 0.133 | 5.782 ± 0.136 |
| S-Aug-KD, mean ± SD | 5.926 ± 0.105 | 5.979 ± 0.032 | 6.531 ± 0.428 | 5.918 ± 0.220 |

## Deliverable B: Change Relative to Clean Input 

| Model | Gamma ΔMAE | Blur ΔMAE | Depth-Attenuation ΔMAE |
|---|---:|---:|---:|
| PanEcho teacher | −0.082 | +2.831 | +0.105 |
| S-Sup, mean ± SD | +0.436 ± 0.500 | +1.460 ± 1.201 | +0.112 ± 0.095 |
| S-KD, mean ± SD | +0.287 ± 0.241 | +1.228 ± 1.047 | +0.071 ± 0.071 |
| S-Aug, mean ± SD | +0.140 ± 0.064 | +0.580 ± 0.056 | +0.023 ± 0.064 |
| S-Aug-KD, mean ± SD | +0.054 ± 0.078 | +0.606 ± 0.456 | −0.008 ± 0.119 |

Explanation: Gaussian blur caused the largest increase in MAE overall. S-Aug had the lowest mean MAE among student models in every condition. Under gamma and depth attenuation, S-Aug-KD showed smaller increases from its higher clean MAE, while S-Aug still had lower MAE under both perturbations.

## Deliverable C: Final Efficiency Summary

Same GPU and precision as job 430; batch = 1 video; 50 timed pilot_eval videos after 5 warm-up videos. Medians are shown, with P95 in parentheses.

| System | Input Protocol | Parameters | Single-Clip Latency | Video-Level Latency | Peak Memory |
|---|---|---:|---:|---:|---:|
| PanEcho teacher | Controlled Phase 1 input: 4 × 16 frames, 224 × 224, one 4-clip forward | 42,019,681 | 25.49 ms | 145.5 ms (147.7) | 979.8 MiB |
| X3D-M student | Controlled Phase 1 input: 4 × 16 frames, 224 × 224, one 4-clip forward | 2,976,723 | 10.46 ms | 77.8 ms (81.4) | 518.6 MiB |
| Phase 0 R(2+1)D reference | Original Phase 0 input: all 32-frame, period-2 clips at 112 × 112 (median 110 clips/video, range 57–530), blocks of 20 | 31,300,638 | 37.15 ms | 5,192.3 ms (9,814.9) | 6,239.3 MiB |



| Comparison | Mean MAE Difference | 95% Paired Bootstrap CI |
|---|---:|---|
| S-Aug-KD − S-Aug (clean) | +0.167 pp | [+0.007, +0.322] |

These CIs reflect pilot-sample uncertainty for the 3-seed average. Training-seed variability is reported separately as SD.

---

# Phase 1 Questions

1. **Does PanEcho provide useful EF supervision under the controlled Phase 1 input protocol?**
Only to a limited extent. Without any fine-tuning it predicts EF reasonably (pilot MAE 5.78 pp, RMSE 7.87), confirming that the teacher works. However, it is no more accurate than a supervised X3D-M (5.93 ± 0.18), is slightly less accurate than S-Aug (5.76 ± 0.08), and has the largest average blur degradation of any system (+2.83 pp).

2. **Is X3D-M meaningfully cheaper to run than the teacher and the existing Phase 0 EF system?**
Yes. X3D-M is **much cheaper** to run than PanEcho:, X3D-M has about one-fourteenth as many parameters. In our tests, it was about 2.4 times as fast per clip and 1.9 times as fast per video.

3. **Does standard KD improve X3D-M compared with supervised training alone?**
The benefit of standard KD depended on the input condition. On clean inputs, S-KD had a mean MAE 0.08 pp lower than S-Sup and performed better in two of three seeds. The paired 95% bootstrap CI [−0.232, +0.068] pp included zero, so the clean-input result was inconclusive. Under gamma and blur, S-KD reduced mean MAE by 0.23 pp and 0.32 pp, respectively, and both confidence intervals were entirely below zero.

4. **After ordinary augmentation is included, does KD still provide an additional benefit?** Augmentation alone gave the lower mean MAE across all evaluated input conditions. On clean inputs, S-Aug-KD had higher MAE than S-Aug in all three seeds. The average increase was 0.167 pp, with a paired 95% bootstrap CI of [+0.007, +0.322] pp. S-Aug-KD also had higher mean MAE under gamma, blur, and depth attenuation. Among those three stress tests, only the blur comparison had a confidence interval entirely above zero.

5. **Are any gains consistent across three training seeds?**
   - No KD gain is consistent: S-KD beat S-Sup on clean input for 2 of 3 seeds, and S-Aug-KD was worse than S-Aug for 3 of 3.
   - The most consistent pattern is the augmentation effect under stress: S-Aug had lower blur and depth MAE than S-Sup for all 3 seeds, but lower clean MAE for only 2 of 3.
   - S-Aug also had the smallest seed SD (clean 0.08 pp).
   
6. **Do the same conclusions remain visible under the fixed input-change stress tests?**
   - Yes, with one nuance: KD's only benefit appears when no augmentation is used, and it is smaller than the benefit of augmentation.
   - S-Aug has the lowest mean MAE of all student models under clean, gamma, blur and depth conditions. Adding KD to augmentation does not reduce perturbed MAE.

### Phase 1 Decision

| Question | Result | Decision |
|---|---|---|
| Teacher provides a useful signal | Yes, limited: the teacher works (MAE 5.78) but is not more accurate than the trained students and is blur-sensitive | Keep it as a reference; do not build on PanEcho EF distillation |
| Student has a meaningful efficiency advantage | Yes | X3D-M remains the compact model of choice |
| KD improves over matched supervised baseline | Inconclusive (clean); some robustness benefit without augmentation | Not sufficient to justify KD |
| KD adds value beyond ordinary augmentation | No | Prefer S-Aug |
| Improvement is consistent across 3 seeds | No (for KD). | Augmentation: yes under blur/depth stress, 2 of 3 seeds on clean input  |
| A meaningful gap remains for Phase 2 research | Not Now ||

## Limitations

- **Evaluation scope.** We evaluated performance on 644 videos from the official validation split and held out the official test set.
- **Synthetic perturbations.** The stress tests used synthetic gamma changes, blur, and depth attenuation. The test gamma of 1.5 was outside the training augmentation range of 0.8–1.25. Cross-hospital and cross-device performance requires separate validation.
- **Input protocol.** We compared PanEcho and X3D-M using four 16-frame clips per video. The accuracy comparison applies to this shared input protocol.
- **Inference timing.** We measured latency on a single NVIDIA GB10 GPU using FP32. The reported speedups reflect this hardware and the tested input protocols.
