import cv2
import numpy as np
import torch
from pathlib import Path
from scripts.models.base import Detection
from scripts.config import CLASS_NAMES

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def draw_and_save_preview(
    image_tensor: torch.Tensor,
    pred_detection: Detection,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    save_path: str | Path,
    image_size: int,
) -> None:
    """
    Denormalize the image tensor, draw Ground Truth (Green) and Predictions (Red), and save to disk.
    target_boxes should be absolute xyxy.
    """
    arr = image_tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    
    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    # Ground Truth context (Green)
    for t_box, t_lbl in zip(target_boxes, target_labels):
        x1, y1, x2, y2 = t_box.cpu().numpy().astype(int)
        lbl_idx = int(t_lbl.item())
        name = CLASS_NAMES[lbl_idx] if lbl_idx < len(CLASS_NAMES) else str(lbl_idx)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"GT: {name}", (x1, max(y1-5, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Predictions (Red)
    for p_box, p_score, p_lbl in zip(pred_detection.boxes, pred_detection.scores, pred_detection.labels):
        # p_box is normalized [0, 1]
        x1, y1, x2, y2 = (p_box * image_size).astype(int)
        lbl_idx = int(p_lbl.item())
        name = CLASS_NAMES[lbl_idx] if lbl_idx < len(CLASS_NAMES) else str(lbl_idx)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_bgr, f"Pred: {name} {p_score:.2f}", (x1, min(y2+15, image_size-5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img_bgr)
