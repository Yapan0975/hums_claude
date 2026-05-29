"""RQ1 closed-set mIoU + ECE evaluation runner.

Used by ``scripts/eval_rq1.sh`` and ``scripts/cloud_train.sh``. Streams a
SemanticKITTI sequence through one of the supported systems and writes a
JSON results file.

CLI::

    python -m evidlife_map.eval.rq1 \
        --config configs/evidlife_kitti.yaml \
        --checkpoint runs/m1_v1.ckpt \
        --system evidlife_map \
        --sequences 08 \
        --output runs/rq1_evidlife.json

Supported ``--system``:
    evidlife_map   M1 EDL head + EvidenceAccumulator
    r2_reimpl      R2 pipeline with argmax-Bayes (jiao2024r2 baseline)
    convbki        ConvBKI wrapper (wilson2024convbki)
    sbki           S-BKI wrapper (gan2020sbki)
    kimera_semantics  Kimera-Semantics wrapper
    nvblox_bare    NvBlox without semantic head (geometric upper bound)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor

from evidlife_map.baselines.r2_reimpl.r2_pipeline import R2Pipeline, R2PipelineConfig
from evidlife_map.data.semantic_kitti import (
    SEMANTIC_KITTI_NUM_CLASSES,
    SemanticKITTIDataset,
)
from evidlife_map.eval.ece import ECECalculator
from evidlife_map.eval.miou import IoUTracker
from evidlife_map.m1_evidential.edl_head import EDLHead
from evidlife_map.m1_evidential.fusion import EvidenceAccumulator


SUPPORTED_SYSTEMS = {
    "evidlife_map", "r2_reimpl", "convbki", "sbki",
    "kimera_semantics", "nvblox_bare",
}


class _PerPointRunner:
    """Per-point predictor + probability emitter.

    Each system is mapped to a callable ``(points_xyz_m, labels) -> (pred, prob)``
    so the same IoU / ECE machinery can grade all of them uniformly.
    """

    def __init__(self, system: str, num_classes: int, device: torch.device, checkpoint: Path | None) -> None:
        self.system = system
        self.num_classes = num_classes
        self.device = device

        if system == "evidlife_map":
            self.edl = EDLHead(
                in_features=num_classes,
                num_classes=num_classes,
                activation="softplus",
            ).to(device)
            self.acc = EvidenceAccumulator(
                num_classes_plus_one=num_classes + 1, device=device,
            )
            if checkpoint is not None and checkpoint.exists():
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                self.edl.load_state_dict(state)
        elif system in ("r2_reimpl", "convbki", "sbki", "kimera_semantics", "nvblox_bare"):
            self.r2 = R2Pipeline(R2PipelineConfig(num_classes=num_classes, device=str(device)))
            # NOTE: convbki / sbki / kimera_semantics wrappers are TODO (W2-W3);
            # the placeholder reuses the R2 pipeline so the eval harness is
            # exercised end-to-end even before the wrappers are integrated.
        else:
            raise ValueError(f"unknown system {system!r}")

    def predict(self, points: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(pred, prob)``: per-point argmax + per-point class probabilities."""
        if self.system == "evidlife_map":
            # M1 head needs C-dim per-point features. As a stand-in for a real
            # semantic backbone, we still go through the R2 MLP semantic head
            # so RQ1 stays comparable; the EDL head sits on top of those probs.
            r2 = R2Pipeline(R2PipelineConfig(num_classes=self.num_classes, device=str(self.device)))
            probs = r2.stage3_semantic_head(points)  # (N, C)
            with torch.no_grad():
                alpha, _ = self.edl(probs.to(self.device))
            # Drop the unknown channel for the closed-set IoU comparison.
            mean = (alpha[..., :self.num_classes] / alpha.sum(dim=-1, keepdim=True))
            pred = mean.argmax(dim=-1).cpu()
            prob = mean.cpu()
            return pred, prob
        # All others: argmax over R2 stage3 probabilities.
        probs = self.r2.stage3_semantic_head(points)
        pred = probs.argmax(dim=-1)
        return pred, probs


def evaluate_sequence(
    dataset: SemanticKITTIDataset,
    runner: _PerPointRunner,
    *,
    num_classes: int,
    log_every: int = 50,
) -> dict[str, object]:
    iou = IoUTracker(num_classes=num_classes, ignore_index=-100)
    ece = ECECalculator(num_bins=15)
    t0 = time.perf_counter()
    n = len(dataset)
    for i in range(n):
        entry = dataset[i]
        pred, prob = runner.predict(entry.points_xyz_m, entry.labels)
        iou.update(pred, entry.labels)
        # ECE: drop ignored labels.
        valid = entry.labels >= 0
        if valid.any():
            ece.update(prob[valid].cpu(), entry.labels[valid].cpu())
        if (i + 1) % log_every == 0:
            partial = iou.compute()
            elapsed = time.perf_counter() - t0
            print(f"  [{i+1}/{n}] frames, partial mIoU={partial['miou']:.4f}, "
                  f"elapsed={elapsed:.0f}s ({(i+1)/elapsed:.1f} fps)")
    elapsed = time.perf_counter() - t0
    metric = iou.compute()
    metric["ece"] = ece.compute()
    metric["latency_s"] = elapsed
    metric["fps"] = n / elapsed if elapsed > 0 else 0.0
    return metric


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None,
                    help="(Reserved) YAML config — for now CLI args override.")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--system", required=True, choices=sorted(SUPPORTED_SYSTEMS))
    ap.add_argument("--root", type=Path, required=True,
                    help="SemanticKITTI dataset root.")
    ap.add_argument("--sequences", default="08",
                    help="Comma-separated sequence list (e.g. 08,11,12).")
    ap.add_argument("--num-classes", type=int, default=SEMANTIC_KITTI_NUM_CLASSES)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--output", type=Path, required=True,
                    help="JSON output file.")
    ap.add_argument("--limit-frames", type=int, default=None,
                    help="Cap frame count per sequence (smoke testing).")
    args = ap.parse_args()

    root = args.root
    if (root / "dataset" / "sequences").exists():
        root = root / "dataset"
        print(f"  auto-located dataset/ subdir at {root}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    print(f"system={args.system}, sequences={sequences}, device={device}")

    runner = _PerPointRunner(args.system, args.num_classes, device, args.checkpoint)

    results: dict[str, dict[str, object]] = {}
    for seq in sequences:
        print(f"==> evaluating sequence {seq}")
        ds = SemanticKITTIDataset(
            root=root,
            sequences=[seq],
            synthetic=False,
        )
        if args.limit_frames is not None:
            # Slice the index to first N frames (simple view).
            ds._frame_index = ds._frame_index[:args.limit_frames]  # noqa: SLF001
        print(f"  loaded {len(ds)} frames from seq {seq}")
        metric = evaluate_sequence(ds, runner, num_classes=args.num_classes)
        print(f"  seq {seq}: mIoU={metric['miou']:.4f}, "
              f"ECE={metric['ece']:.4f}, fps={metric['fps']:.2f}")
        results[seq] = metric

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "system": args.system,
        "device": str(device),
        "num_classes": args.num_classes,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "sequences": results,
    }
    args.output.write_text(json.dumps(out, indent=2, default=str))
    print(f"==> wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
