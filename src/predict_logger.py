"""
Disk-based prediction logger.

Each model writes its predictions image-by-image to a JSONL file:
    outputs/<arch>/predictions_<split>.jsonl

One line per image:
    {"image_id": "abc123", "boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...]}

The stacking step reads all per-model JSONL files and fuses them with WBF —
without ever loading more than one model into memory at a time.
"""

from __future__ import annotations


import json
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError:
    MeanAveragePrecision = None  # type: ignore

from src.config import DataConfig, EnsembleConfig
from src.data.dataset import build_dataloader
from src.models.base import BaseDetector, Detection


# ---------------------------------------------------------------------------
# JSONL writer / reader
# ---------------------------------------------------------------------------


class PredictionWriter:
    """Appends one Detection per line to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def write(self, image_id: str, det: Detection) -> None:
        record = {
            "image_id": image_id,
            "boxes": det.boxes.tolist(),
            "scores": det.scores.tolist(),
            "labels": det.labels.tolist(),
        }
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "PredictionWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def iter_predictions(path: str | Path) -> Iterator[tuple[str, Detection]]:
    """Yield (image_id, Detection) from a JSONL prediction file."""
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            det = Detection(
                boxes=np.array(rec["boxes"], dtype=np.float32),
                scores=np.array(rec["scores"], dtype=np.float32),
                labels=np.array(rec["labels"], dtype=np.int32),
            )
            yield rec["image_id"], det


def load_predictions(path: str | Path) -> dict[str, Detection]:
    """Load an entire JSONL prediction file into a dict."""
    return {img_id: det for img_id, det in iter_predictions(path)}


# ---------------------------------------------------------------------------
# Per-model prediction runner (one image at a time)
# ---------------------------------------------------------------------------


def predict_and_log(
    model: BaseDetector,
    data_cfg: DataConfig,
    split: Literal["train", "val", "test"],
    output_path: str | Path,
    score_threshold: float = 0.0,
    batch_size: int = 1,
    prepared_dataset_root: str | Path | None = None,
) -> Path:
    """
    Run model inference over the entire split, writing each prediction
    immediately to disk.  Uses batch_size=1 by default to minimise peak RAM.

    prepared_dataset_root : path to the output of prepare_dataset.py
        (e.g. "data/processed").  When provided, images are loaded
        from pre-converted 16-bit PNGs instead of raw DICOMs.

    Returns the path to the written JSONL file.
    """
    output_path = Path(output_path)

    import copy

    cfg_1 = copy.copy(data_cfg)
    cfg_1.batch_size = batch_size
    cfg_1.num_workers = min(data_cfg.num_workers, 2)

    output_format = "torchvision"
    loader = build_dataloader(
        root=data_cfg.root,
        split=split,
        cfg=cfg_1,
        output_format=output_format,
        shuffle=False,
        prepared_dataset_root=prepared_dataset_root,
    )

    total = len(loader.dataset)  # type: ignore[arg-type]
    print(f"  Predicting {total} images → {output_path}")

    with PredictionWriter(output_path) as writer:
        for images, targets in tqdm(loader, desc="predict", total=len(loader)):
            if isinstance(images, torch.Tensor):
                image_list = [images[i] for i in range(images.shape[0])]
            else:
                image_list = list(images)

            detections = model.predict(image_list, image_size=data_cfg.image_size)

            for det, target in zip(detections, targets):
                img_id = target["image_id"]
                filtered = det.filter_by_score(score_threshold)
                writer.write(img_id, filtered)

    return output_path


# ---------------------------------------------------------------------------
# Stacking from saved prediction files
# ---------------------------------------------------------------------------


def stack_predictions(
    pred_files: list[str | Path],
    output_path: str | Path,
    cfg: EnsembleConfig | None = None,
    weights: list[float] | None = None,
) -> Path:
    """
    Fuse per-model JSONL prediction files using WBF — no models in memory.

    pred_files : one JSONL file per model, in the same order as `weights`
    output_path: where to write the fused JSONL file
    """
    from src.stacking.ensemble import StackingEnsemble
    from src.config import EnsembleConfig

    if cfg is None:
        cfg = EnsembleConfig()
    if weights is not None:
        cfg.weights = weights

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all image IDs (union across all files)
    all_ids: list[str] = []
    seen: set[str] = set()
    for pf in pred_files:
        for img_id, _ in iter_predictions(pf):
            if img_id not in seen:
                all_ids.append(img_id)
                seen.add(img_id)

    # Load all predictions into memory — only boxes/scores/labels, not images
    per_model: list[dict[str, Detection]] = [load_predictions(pf) for pf in pred_files]

    # Dummy ensemble with no models — we call _fuse_single directly
    dummy_ensemble = StackingEnsemble.__new__(StackingEnsemble)
    dummy_ensemble.models = []
    dummy_ensemble.cfg = cfg

    print(
        f"  Fusing {len(all_ids)} images from {len(pred_files)} models → {output_path}"
    )

    with PredictionWriter(output_path) as writer:
        for img_id in tqdm(all_ids, desc="fuse"):
            img_preds = [
                model_preds.get(img_id, Detection()) for model_preds in per_model
            ]
            fused = dummy_ensemble._fuse_single(img_preds)
            writer.write(img_id, fused)

    return output_path


# ---------------------------------------------------------------------------
# mAP evaluation from a prediction JSONL file
# ---------------------------------------------------------------------------


def evaluate_predictions(
    pred_file: str | Path,
    data_cfg: DataConfig,
    split: Literal["train", "val", "test"] = "val",
) -> dict:
    """
    Compute mAP from a saved JSONL prediction file against ground truth.
    Requires torchmetrics.
    """
    if MeanAveragePrecision is None:
        raise ImportError("pip install torchmetrics")

    preds_dict = load_predictions(pred_file)

    loader = build_dataloader(
        root=data_cfg.root,
        split=split,
        cfg=data_cfg,
        output_format="torchvision",
        shuffle=False,
    )

    metric: Any = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    for _, targets in loader:
        for target in targets:
            img_id = target["image_id"]
            det = preds_dict.get(img_id, Detection())

            pred_fmt = {
                "boxes": torch.tensor(det.boxes * data_cfg.image_size),
                "scores": torch.tensor(det.scores),
                "labels": torch.tensor(det.labels, dtype=torch.long),
            }
            gt_fmt = {
                "boxes": target["boxes"],
                "labels": target["labels"],
            }
            metric.update(preds=[pred_fmt], target=[gt_fmt])

    result: dict = metric.compute()
    return {k: float(v.item()) if hasattr(v, "item") else v for k, v in result.items()}
