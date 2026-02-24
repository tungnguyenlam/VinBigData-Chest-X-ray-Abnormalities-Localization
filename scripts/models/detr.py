from __future__ import annotations

from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor

from scripts.config import CLASS_NAMES, NUM_CLASSES, DataConfig, ModelConfig
from scripts.models.base import BaseDetector, Detection


# Map backbone_size string to HuggingFace model checkpoint
_DETR_CHECKPOINTS: dict[str, str] = {
    "resnet-50": "facebook/detr-resnet-50",
    "resnet-101": "facebook/detr-resnet-101",
    # backbone_size aliases used by HardwarePreset
    "resnet50": "facebook/detr-resnet-50",
    "resnet101": "facebook/detr-resnet-101",
    "n": "facebook/detr-resnet-50",
    "s": "facebook/detr-resnet-50",
    "m": "facebook/detr-resnet-50",
    "l": "facebook/detr-resnet-101",
    "x": "facebook/detr-resnet-101",
}


class DETRDetector(BaseDetector):
    """
    HuggingFace DETR wrapper for VinBigData localization.

    Expects DataLoader in torchvision format:
        images : list[Tensor[C,H,W]] — normalized float tensors
        targets: list[dict{"boxes": Tensor[N,4] xyxy absolute,
                            "labels": Tensor[N] int64}]

    DETR uses a Hungarian matcher and does not need NMS — predictions are
    post-processed by the HuggingFace processor.
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

        checkpoint = _DETR_CHECKPOINTS.get(
            model_cfg.backbone_size, "facebook/detr-resnet-50"
        )

        # Load config, override num_labels to our dataset
        hf_cfg = DetrConfig.from_pretrained(checkpoint)
        hf_cfg.num_labels = NUM_CLASSES
        hf_cfg.id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
        hf_cfg.label2id = {name: i for i, name in enumerate(CLASS_NAMES)}

        self.model = DetrForObjectDetection.from_pretrained(
            checkpoint,
            config=hf_cfg,
            ignore_mismatched_sizes=True,
        )
        self.model.to(device=self.device)

        # Processor handles pixel normalization and target formatting for HF
        self.processor = DetrImageProcessor.from_pretrained(checkpoint)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
            train_loss = self._train_epoch(train_loader, optimizer, scaler)
            val_loss = self._val_epoch(val_loader, epoch)
            scheduler.step()

            print(
                f"[DETR] Epoch {epoch}/{self.model_cfg.epochs} — "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                self.save(best_path)

    def _prepare_batch(
        self,
        images: list[torch.Tensor],
        targets: list[dict],
        image_size: int,
    ) -> tuple[dict, list[dict]]:
        """
        Convert list of CHW tensors + torchvision-style targets to the format
        expected by DetrForObjectDetection.

        DETR expects:
            pixel_values  : Tensor[B, C, H, W]
            pixel_mask    : Tensor[B, H, W]
            labels        : list of dicts with keys:
                              class_labels (int64), boxes (float cxcywh normalized)
        """
        pixel_values = torch.stack(images, dim=0).to(self.device)
        pixel_mask = torch.ones(
            pixel_values.shape[0],
            pixel_values.shape[2],
            pixel_values.shape[3],
            dtype=torch.long,
            device=self.device,
        )

        hf_targets = []
        for t in targets:
            boxes_xyxy = t["boxes"]  # absolute xyxy
            labels = t["labels"]

            if len(boxes_xyxy) == 0:
                hf_targets.append(
                    {
                        "class_labels": torch.zeros(
                            0, dtype=torch.long, device=self.device
                        ),
                        "boxes": torch.zeros(
                            (0, 4), dtype=torch.float32, device=self.device
                        ),
                    }
                )
                continue

            # Normalize and convert to cx cy w h
            cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / image_size
            cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / image_size
            bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / image_size
            bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / image_size
            cxcywh = torch.stack([cx, cy, bw, bh], dim=1).to(self.device)

            hf_targets.append(
                {
                    "class_labels": labels.to(self.device),
                    "boxes": cxcywh,
                }
            )

        encoding = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return encoding, hf_targets

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        img_size = self.data_cfg.image_size

        for images, targets in loader:
            # Skip if batch has no annotated images
            if all(len(t["boxes"]) == 0 for t in targets):
                continue

            encoding, hf_targets = self._prepare_batch(images, targets, img_size)

            optimizer.zero_grad()
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.model_cfg.amp and self.device.type == "cuda",
            ):
                outputs = self.model(**encoding, labels=hf_targets)
                loss = outputs.loss

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
        self.model.train()  # keep in train mode to compute DETR loss
        total_loss = 0.0
        img_size = self.data_cfg.image_size
        saved_preview = False

        with torch.no_grad():
            for images, targets in loader:
                if all(len(t["boxes"]) == 0 for t in targets):
                    continue

                if not saved_preview:
                    saved_preview = True
                    self.model.eval()
                    valid_idx = next(
                        i for i, t in enumerate(targets) if len(t["boxes"]) > 0
                    )
                    preds = self.predict([images[valid_idx]], image_size=img_size)
                    self.model.train()

                    from scripts.visualize import draw_and_save_preview

                    save_path = (
                        self.output_dir / "val_previews" / f"epoch_{epoch:03d}.jpg"
                    )
                    draw_and_save_preview(
                        image_tensor=images[valid_idx],
                        pred_detection=preds[0],
                        target_boxes=targets[valid_idx]["boxes"],
                        target_labels=targets[valid_idx]["labels"],
                        save_path=save_path,
                        image_size=img_size,
                    )

                encoding, hf_targets = self._prepare_batch(images, targets, img_size)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.model_cfg.amp and self.device.type == "cuda",
                ):
                    outputs = self.model(**encoding, labels=hf_targets)
                    loss = outputs.loss

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
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = list(images)

        pixel_values = torch.stack(image_list, dim=0).to(self.device)
        pixel_mask = torch.ones(
            pixel_values.shape[0],
            pixel_values.shape[2],
            pixel_values.shape[3],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.tensor(
            [[image_size, image_size]] * len(image_list),
            dtype=torch.float32,
        )
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.0,  # return all predictions; caller filters by score
        )

        detections: list[Detection] = []
        for r in results:
            boxes = r["boxes"].cpu().numpy()  # absolute xyxy
            scores = r["scores"].cpu().numpy()
            labels = r["labels"].cpu().numpy().astype(np.int32)

            # Normalize to [0, 1]
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
        self.model.save_pretrained(str(path.with_suffix("")))
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        hf_dir = path.with_suffix("")
        if hf_dir.is_dir():
            self.model = DetrForObjectDetection.from_pretrained(str(hf_dir))
            self.model.to(device=self.device)
        else:
            state = torch.load(str(path), map_location=self.device)
            self.model.load_state_dict(state)
