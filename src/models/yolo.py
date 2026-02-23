from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import NUM_CLASSES, DataConfig, ModelConfig
from src.models.base import BaseDetector, Detection


ModelSize = Literal["n", "s", "m", "l", "x"]

# Map backbone_size to YOLOv8 model name
_YOLO_WEIGHTS: dict[str, str] = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}


class YOLODetector(BaseDetector):
    """
    YOLOv8 wrapper for VinBigData localization.

    Training uses the Ultralytics native training loop which expects:
      - images in a flat directory
      - YOLO-format .txt label files in a sibling 'labels/' directory
      - a dataset YAML file

    Inference returns Detection objects with normalized xyxy boxes.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
    ) -> None:
        from ultralytics import YOLO

        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.output_dir = model_cfg.output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

        size = model_cfg.backbone_size if model_cfg.backbone_size in _YOLO_WEIGHTS else "s"
        weights = _YOLO_WEIGHTS[size]
        self.model = YOLO(weights)

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _prepare_yolo_dataset(self) -> Path:
        """
        Delegates to src.data.prepare_dataset.prepare_dataset so that the
        on-disk PNG + label layout is shared with Faster R-CNN and DETR.

        Returns the path to dataset.yaml expected by Ultralytics.
        """
        from src.data.prepare_dataset import prepare_dataset

        dataset_root = prepare_dataset(
            data_cfg=self.data_cfg,
            output_root=self.model_cfg.output_path.parent,
        )
        return dataset_root / "dataset.yaml"

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def train_model(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
    ) -> None:
        """
        Train YOLOv8 using Ultralytics native loop.
        train_loader / val_loader are accepted for API compatibility but unused —
        YOLO reads data directly from the prepared on-disk dataset.
        """
        yaml_path = self._prepare_yolo_dataset()

        self.model.train(
            data=str(yaml_path),
            epochs=self.model_cfg.epochs,
            imgsz=self.data_cfg.image_size,
            batch=self.data_cfg.batch_size,
            workers=self.data_cfg.num_workers,
            device=self.model_cfg.device,
            amp=self.model_cfg.amp,
            lr0=self.model_cfg.lr,
            weight_decay=self.model_cfg.weight_decay,
            warmup_epochs=self.model_cfg.warmup_epochs,
            project=str(self.output_dir),
            name="train",
            exist_ok=True,
            verbose=True,
            cache="disk",
        )

        # Point model to best weights
        best = self.output_dir / "train" / "weights" / "best.pt"
        if best.exists():
            self.load(best)

    def predict(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        image_size: int = 640,
    ) -> list[Detection]:
        """
        Run inference.

        Accepts either:
          - a list of CHW float tensors (normalized, from DataLoader)
          - a NCHW batch tensor
          - a list of file path strings / numpy arrays
        """
        import cv2

        _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        _IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if isinstance(images, torch.Tensor):
            image_list: list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = list(images)

        # Ultralytics native inference accepts uint8 BGR numpy arrays.
        # If we received normalized float tensors from the dataloader, undo
        # the ImageNet normalization and convert back to uint8 HWC BGR.
        converted: list = []
        for img in image_list:
            if isinstance(img, torch.Tensor):
                arr = img.cpu().numpy()          # CHW float
                arr = arr.transpose(1, 2, 0)     # HWC
                arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                converted.append(arr)
            else:
                converted.append(img)

        results = self.model(converted, imgsz=image_size, verbose=False)

        detections: list[Detection] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                detections.append(Detection())
                continue

            boxes_xyxy = r.boxes.xyxyn.cpu().numpy()  # normalized xyxy
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(np.int32)

            detections.append(Detection(boxes=boxes_xyxy, scores=scores, labels=labels))

        return detections

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(path))

    def export(self, format: str = "onnx", path: str | Path | None = None) -> Path:
        """Export model to ONNX / TorchScript / etc."""
        out = self.model.export(format=format)
        if path is not None:
            shutil.move(str(out), str(path))
            return Path(path)
        return Path(out)
