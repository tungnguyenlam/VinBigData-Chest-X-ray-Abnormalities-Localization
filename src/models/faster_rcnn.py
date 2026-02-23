from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.config import NUM_CLASSES, DataConfig, ModelConfig
from src.models.base import BaseDetector, Detection


BackboneSize = Literal["resnet50", "resnet50v2", "resnet101"]


def _build_model(backbone_size: str, num_classes: int) -> FasterRCNN:
    """
    Build a Faster R-CNN model with COCO-pretrained backbone, custom head.

    torchvision detection models use 1-indexed labels: class 0 = background,
    classes 1..num_classes = actual categories.  The head therefore needs
    (num_classes + 1) output slots.
    """
    num_classes_with_bg = num_classes + 1  # +1 for background (index 0)

    if backbone_size == "resnet101":
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights="ResNet101_Weights.DEFAULT",
            trainable_layers=3,
        )
        # FasterRCNN constructor takes num_classes INCLUDING background
        model = FasterRCNN(backbone, num_classes=num_classes_with_bg)
        return model

    if backbone_size == "resnet50v2":
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    else:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the pre-trained COCO head with one sized for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


class FasterRCNNDetector(BaseDetector):
    """
    Faster R-CNN wrapper for VinBigData localization.

    Uses torchvision detection API; expects DataLoader in torchvision format:
        images : list[Tensor[C,H,W]]
        targets: list[dict{"boxes": Tensor[N,4] xyxy absolute,
                            "labels": Tensor[N] int64}]
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.output_dir = model_cfg.output_path
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            model_cfg.device
            if torch.cuda.is_available() or model_cfg.device == "cpu"
            else "cpu"
        )
        self.model = _build_model(model_cfg.backbone_size, NUM_CLASSES).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.model_cfg.lr,
            weight_decay=self.model_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.model_cfg.epochs,
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=self.model_cfg.amp and self.device.type == "cuda"
        )

        best_loss = float("inf")
        best_path = self.output_dir / "best.pt"

        for epoch in range(1, self.model_cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, scaler, epoch)
            val_loss = self._val_epoch(val_loader, epoch)
            scheduler.step()

            print(
                f"[FasterRCNN] Epoch {epoch}/{self.model_cfg.epochs} — "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                self.save(best_path)

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        epoch: int,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for images, targets in loader:
            images = [img.to(self.device) for img in images]
            targets_dev = [
                {
                    "boxes": t["boxes"].to(self.device),
                    # torchvision expects 1-indexed labels (0 = background)
                    "labels": (t["labels"] + 1).to(self.device),
                }
                for t in targets
                if len(t["boxes"]) > 0
            ]
            images_with_ann = [
                img for img, t in zip(images, targets) if len(t["boxes"]) > 0
            ]

            if not images_with_ann:
                continue

            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.model_cfg.amp and self.device.type == "cuda",
            ):
                loss_dict = self.model(images_with_ann, targets_dev)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            if self.model_cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.model_cfg.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def _val_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()  # keep in train mode to compute losses
        total_loss = 0.0
        saved_preview = False

        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(self.device) for img in images]
                targets_dev = [
                    {
                        "boxes": t["boxes"].to(self.device),
                        "labels": (t["labels"] + 1).to(self.device),
                    }
                    for t in targets
                    if len(t["boxes"]) > 0
                ]
                images_with_ann = [
                    img for img, t in zip(images, targets) if len(t["boxes"]) > 0
                ]

                if not images_with_ann:
                    continue

                if not saved_preview:
                    saved_preview = True
                    self.model.eval()
                    preds = self.predict(
                        [images_with_ann[0]], image_size=self.data_cfg.image_size
                    )
                    self.model.train()

                    from src.visualize import draw_and_save_preview

                    save_path = (
                        self.output_dir / "val_previews" / f"epoch_{epoch:03d}.jpg"
                    )
                    orig_target = [t for t in targets if len(t["boxes"]) > 0][0]
                    draw_and_save_preview(
                        image_tensor=images_with_ann[0],
                        pred_detection=preds[0],
                        target_boxes=orig_target["boxes"],
                        target_labels=orig_target["labels"],
                        save_path=save_path,
                        image_size=self.data_cfg.image_size,
                    )

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.model_cfg.amp and self.device.type == "cuda",
                ):
                    loss_dict = self.model(images_with_ann, targets_dev)
                    loss = sum(loss_dict.values())

                total_loss += loss.item()

        return total_loss / max(1, len(loader))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        images: list[torch.Tensor] | torch.Tensor,
        image_size: int = 640,
    ) -> list[Detection]:
        self.model.eval()

        if isinstance(images, torch.Tensor):
            image_list = [images[i].to(self.device) for i in range(images.shape[0])]
        else:
            image_list = [img.to(self.device) for img in images]

        with torch.no_grad():
            outputs = self.model(image_list)

        detections: list[Detection] = []
        for out in outputs:
            boxes = out["boxes"].cpu().numpy()  # absolute xyxy
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy().astype(np.int32)

            # Drop background predictions (label == 0)
            fg_mask = labels > 0
            boxes = boxes[fg_mask]
            scores = scores[fg_mask]
            labels = labels[fg_mask]

            # Shift back to 0-indexed to match the rest of the pipeline
            labels = (labels - 1).astype(np.int32)

            # Normalize boxes to [0, 1]
            if len(boxes) > 0:
                boxes[:, [0, 2]] /= image_size
                boxes[:, [1, 3]] /= image_size
                boxes = np.clip(boxes, 0.0, 1.0)

            detections.append(Detection(boxes=boxes, scores=scores, labels=labels))

        return detections

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
