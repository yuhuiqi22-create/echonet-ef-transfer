#!/bin/bash
#SBATCH -J phase1-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00

# Pilot-evaluation of one student checkpoint (plus the fixed PanEcho teacher).
# Sampling, crop, channel order and normalization are copied from job 429
# (phase1-clean-eval.sbatch); only labels, perturbations and output paths differ.
# Usage: sbatch phase1-eval.sbatch CHECKPOINT [clean|gamma|blur|depth] [smoke|full]
set -euo pipefail
umask 077
"$HOME/envs/panecho-phase1/bin/python" -u - "${1:?Pass a student checkpoint}" "${2:-clean}" "${3:-full}" <<'PY'
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from pytorchvideo.models.hub.x3d import x3d_m
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path.home() / "projects" / "dynamic"))
from phase1_perturb import perturb
import phase1_offline  # noqa: F401  (PanEcho tasks.pkl from verified local copy)


MANIFEST_SHA256 = "3da80a3e5ad9c94b65966c16bcbf700f199b03cbdcc3a23481fb26a8610f6080"
FRAMES = 16
EVAL_CLIPS = 4
BATCH_SIZE = 4
PHASE1 = Path.home() / "projects" / "dynamic" / "output" / "phase1"
PERTURBATION_FILE = PHASE1 / "stress" / "perturbations_frozen.json"


def load_pilot_rows(file_list, manifest_path):
    manifest = manifest_path.read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise RuntimeError("Frozen development manifest checksum mismatch.")

    split = {"model_selection": set(), "pilot_eval": set()}
    for row in csv.DictReader(io.StringIO(manifest.decode("utf-8"))):
        split[row["Phase1Subset"]].add(row["FileName"])
    pilot = split["pilot_eval"]
    selection = split["model_selection"]
    if len(pilot) != 644 or len(selection) != 644 or pilot & selection:
        raise RuntimeError("Expected two disjoint sets of 644 videos.")

    official_val = set()
    labels = {}
    with file_list.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Split"].strip().upper() != "VAL":
                continue
            name = row["FileName"].strip()
            if name in official_val:
                raise RuntimeError(f"Duplicate official VAL video: {name}")
            official_val.add(name)
            if name in pilot:
                ef = float(row["EF"])
                if not math.isfinite(ef) or not 0 <= ef <= 100:
                    raise RuntimeError(f"Invalid EF label for {name}: {ef}")
                labels[name] = ef
    if official_val != pilot | selection or set(labels) != pilot:
        raise RuntimeError("Official VAL videos do not match the frozen development split.")
    return sorted(labels.items())


def load_perturbation(kind):
    if kind == "clean":
        return {"kind": "clean"}
    payload = PERTURBATION_FILE.read_bytes()
    frozen = json.loads(payload)
    if hashlib.sha256(json.dumps(frozen["settings"], sort_keys=True).encode()).hexdigest() != frozen["settings_sha256"]:
        raise RuntimeError("Frozen perturbation settings were modified.")
    return {"kind": kind, **frozen["settings"][kind]}


class PilotClips(Dataset):
    def __init__(self, videos, rows, settings):
        self.videos = videos
        self.rows = rows
        self.settings = settings

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        name, ef = self.rows[index]
        filename = name if name.lower().endswith(".avi") else f"{name}.avi"
        video = self.videos / filename
        capture = cv2.VideoCapture(str(video))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < 1:
                raise RuntimeError(f"Could not open video: {video}")
            # Same deterministic validation sampling and crop as job 429 and training selection.
            starts = np.linspace(0, max(0, frame_count - FRAMES), EVAL_CLIPS, dtype=int)
            clips = []
            for start in starts:
                start = int(start)
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise RuntimeError(f"Could not seek to frame {start} in {video}")
                frames = []
                for _ in range(min(FRAMES, frame_count - start)):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Could not decode clip at frame {start} in {video}")
                    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
                    frames.append(frame[16:240, 16:240])
                frames.extend([frames[-1]] * (FRAMES - len(frames)))
                frames = perturb(np.stack(frames), self.settings)
                bgr = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()
                clips.append(bgr)
            return name, ef, torch.stack(clips)
        finally:
            capture.release()


def load_student(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    expected = {
        "mode": "full", "eval_clips": EVAL_CLIPS,
        "manifest_sha256": MANIFEST_SHA256, "target_scale": 100.0,
    }
    if any(settings.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Checkpoint settings do not match the Phase 1 protocol: {settings}")
    if checkpoint["epoch"] != checkpoint["best_epoch"]:
        raise RuntimeError("The checkpoint was not the best selected epoch when saved.")
    condition = settings.get("condition", "S-Sup")  # S-Sup checkpoints predate the condition key.
    student = x3d_m(pretrained=False)
    head = student.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, 1)
    head.activation = None
    student.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"{condition} seed {settings['seed']} checkpoint: {path.resolve()} (epoch {checkpoint['epoch']})")
    print("Checkpoint settings:", settings)
    return student.cuda().eval().requires_grad_(False), condition, settings["seed"], checkpoint["epoch"]


@torch.inference_mode()
def predict(teacher, student, loader):
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1)
    records = []
    for names, labels, clips in loader:
        teacher_clips, student_clips = [], []
        for clip in clips.unbind(dim=1):
            bgr = clip.to("cuda", dtype=torch.float32, non_blocking=True).div_(255.0)
            # PanEcho's loader retains OpenCV channel order. X3D-M was trained on RGB.
            teacher_ef = teacher(((bgr - mean) / std).contiguous())["EF"]
            student_ef = student(bgr.flip((1,)).sub_(0.45).div_(0.225))
            if teacher_ef.shape != (len(names), 1) or student_ef.shape != (len(names), 1):
                raise RuntimeError("Each model must return one EF per input clip.")
            teacher_clips.append(teacher_ef)
            student_clips.append(student_ef.mul(100.0))

        teacher_clip_ef = torch.cat(teacher_clips, dim=1).cpu()
        student_clip_ef = torch.cat(student_clips, dim=1).cpu()
        teacher_ef = teacher_clip_ef.mean(dim=1).tolist()
        student_ef = student_clip_ef.mean(dim=1).tolist()
        for i, name in enumerate(names):
            records.append((name, labels[i].item(), teacher_ef[i], student_ef[i],
                            *teacher_clip_ef[i].tolist(), *student_clip_ef[i].tolist()))
        if len(records) % 64 == 0 or len(records) == len(loader.dataset):
            print(f"Evaluated {len(records)}/{len(loader.dataset)} videos")
    return records


def metrics(records, condition):
    values = np.asarray([row[1:4] for row in records], dtype=np.float64)
    if not np.isfinite(np.asarray([row[1:] for row in records], dtype=np.float64)).all():
        raise RuntimeError("Non-finite EF in evaluation outputs.")
    errors = values[:, 1:] - values[:, :1]
    mae = np.abs(errors).mean(axis=0)
    rmse = np.sqrt(np.square(errors).mean(axis=0))
    return [(model, len(records), mae[i], rmse[i]) for i, model in enumerate(("PanEcho", condition))]


def main():
    checkpoint_path, kind, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    if kind not in {"clean", "gamma", "blur", "depth"} or mode not in {"smoke", "full"}:
        raise SystemExit("Usage: CHECKPOINT [clean|gamma|blur|depth] [smoke|full]")
    cv2.setNumThreads(0)
    root = Path.home() / "datasets" / "EchoNet-Dynamic"
    rows = load_pilot_rows(root / "FileList.csv", PHASE1 / "splits" / "phase1_dev_split_seed2026.csv")
    if mode == "smoke":
        rows = rows[:8]
    settings = load_perturbation(kind)
    student, condition, seed, epoch = load_student(checkpoint_path)
    teacher_source = Path(torch.hub.get_dir()) / "CarDS-Yale_PanEcho_main"
    teacher = torch.hub.load(
        str(teacher_source), "PanEcho", source="local", pretrained=True, clip_len=FRAMES, tasks=["EF"],
    ).cuda().eval().requires_grad_(False)
    if sum(p.numel() for p in teacher.parameters()) != 42_019_681:
        raise RuntimeError("PanEcho parameter count differs from the verified EF-only model.")

    print("GPU:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
    print("PanEcho checkpoint: CarDS-Yale/PanEcho release v1.0 panecho.pt (EF only)")
    print("Perturbation:", settings)
    print("Split: pilot_eval | Videos:", len(rows), "| Clips per video:", EVAL_CLIPS)
    loader = DataLoader(PilotClips(root / "Videos", rows, settings), batch_size=BATCH_SIZE,
                        num_workers=2, pin_memory=True)
    records = predict(teacher, student, loader)
    if [row[0] for row in records] != [row[0] for row in rows]:
        raise RuntimeError("Evaluated videos do not match the frozen pilot list in order.")
    results = metrics(records, condition)

    tag = condition.lower().replace("-", "_")
    destination = (PHASE1 / "eval" / f"{tag}_seed{seed}"
                   / f"{kind}_{mode}_job{os.environ['SLURM_JOB_ID']}")
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("FileName", "Reference_EF", "PanEcho_EF", "Student_EF",
                         *(f"PanEcho_clip{i}" for i in range(EVAL_CLIPS)),
                         *(f"Student_clip{i}" for i in range(EVAL_CLIPS))))
        writer.writerows(records)
    with (destination / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Model", "Videos", "MAE_pp", "RMSE_pp"))
        writer.writerows(results)
    with (destination / "run.json").open("w") as stream:
        json.dump({"checkpoint": str(checkpoint_path.resolve()), "condition": condition, "seed": seed,
                   "epoch": epoch, "perturbation": settings, "mode": mode, "split": "pilot_eval",
                   "videos": len(records), "clips_per_video": EVAL_CLIPS,
                   "slurm_job_id": os.environ["SLURM_JOB_ID"]}, stream, indent=2)
    for model, count, mae, rmse in results:
        print(f"{model} | {kind} | Videos {count} | MAE {mae:.3f} pp | RMSE {rmse:.3f} pp")
    print("Results directory:", destination)
    print("Phase 1 pilot evaluation completed")


if __name__ == "__main__":
    main()
PY
