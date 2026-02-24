from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pydicom

import src.config as app_config
from src.config import NO_FINDING_CLASS_ID


# ---------------------------------------------------------------------------
# DICOM → numpy
# ---------------------------------------------------------------------------


def _read_dicom_pixels(path: str | Path) -> tuple[np.ndarray, object]:
    """
    Read a DICOM file and return (float32 pixel array after VOI LUT, pydicom dataset).
    Handles MONOCHROME1 inversion so bright=bone/dense tissue consistently.
    """
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    try:
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    if (
        hasattr(ds, "PhotometricInterpretation")
        and ds.PhotometricInterpretation == "MONOCHROME1"
    ):
        arr = arr.max() - arr

    return arr, ds


def dicom_to_array(path: str | Path, apply_clahe: bool = True) -> np.ndarray:
    """Read a DICOM file and return a uint8 3-channel BGR array (H, W, 3)."""
    arr, _ = _read_dicom_pixels(path)

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    arr = arr.astype(np.uint8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr = clahe.apply(arr)

    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr


def dicom_to_3channel_8bit(
    path: str | Path, image_size: int | None = None
) -> np.ndarray:
    """
    Read DICOM, apply VOI LUT, and build a 3-channel 8-bit image for dual-phase training.
    Ch1: Standard Window (baseline)
    Ch2: CLAHE (local contrast)
    Ch3: Unsharp Masking (edge enhancement)
    """
    arr, ds = _read_dicom_pixels(path)

    # Normalize to [0, 1] based on percentiles to handle outliers
    lower_val = np.percentile(arr, 0.5)
    upper_val = np.percentile(arr, 99.5)
    if upper_val > lower_val:
        arr_f32 = np.clip(arr, lower_val, upper_val)
        arr_f32 = (arr_f32 - lower_val) / (upper_val - lower_val)
    else:
        arr_f32 = np.zeros_like(arr, dtype=np.float32)

    if image_size:
        # Resize here to save CPU on CLAHE and Blur
        arr_f32 = cv2.resize(
            arr_f32, (image_size, image_size), interpolation=cv2.INTER_AREA
        )

    # Channel 1: Standard Window (0-255 uint8)
    ch1 = (arr_f32 * 255.0).astype(np.uint8)

    # Channel 2: CLAHE on 16-bit, then down to 8-bit
    arr16 = (arr_f32 * 65535.0).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch2_16 = clahe.apply(arr16)
    ch2 = (ch2_16 / 65535.0 * 255.0).astype(np.uint8)

    # Channel 3: Unsharp Masking
    blurred = cv2.GaussianBlur(arr_f32, (0, 0), 3)
    ch3_f32 = np.clip(arr_f32 * 1.5 - blurred * 0.5, 0.0, 1.0)
    ch3 = (ch3_f32 * 255.0).astype(np.uint8)

    # Stack to (H, W, 3)
    return np.stack([ch1, ch2, ch3], axis=-1)


def load_image(path: str | Path, size: int, apply_clahe: bool = True) -> np.ndarray:
    """
    Load a DICOM or PNG/JPG, resize to (size, size), return uint8 BGR.
    """
    path = Path(path)
    if path.suffix.lower() in (".dicom", ".dcm"):
        arr = dicom_to_array(path, apply_clahe=apply_clahe)
    else:
        # Load directly as uint8 BGR (standard for YOLO/Inference)
        arr = cv2.imread(str(path))
        if arr is None:
            raise FileNotFoundError(f"Could not read image: {path}")

    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_LINEAR)
    return arr


# ---------------------------------------------------------------------------
# Annotation loading & WBF consensus
# ---------------------------------------------------------------------------


def load_annotations(csv_path: str | Path) -> pd.DataFrame:
    """Load train.csv, drop 'No finding' rows, return clean dataframe."""
    df = pd.read_csv(str(csv_path))
    df = df[df["class_id"] != NO_FINDING_CLASS_ID].copy()
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max"])
    df[["x_min", "y_min", "x_max", "y_max"]] = df[
        ["x_min", "y_min", "x_max", "y_max"]
    ].astype(float)
    return df


def get_image_dims(dicom_path: str | Path) -> tuple[int, int]:
    """Return (height, width) of a DICOM image without decoding pixels."""
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
    return int(ds.Rows), int(ds.Columns)


def wbf_single_image(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    iou_thr: float = 0.5,
    skip_box_thr: float = 0.0001,
    weights: Optional[list[float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Weighted Box Fusion on boxes from multiple annotators / models for a
    single image.  All boxes must be normalized to [0, 1].

    Returns:
        boxes  : (N, 4) float32 normalized xyxy
        scores : (N,)   float32
        labels : (N,)   int32
    """
    from ensemble_boxes import weighted_boxes_fusion

    if not boxes_list:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int32),
        )

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    return (
        fused_boxes.astype(np.float32),
        fused_scores.astype(np.float32),
        fused_labels.astype(np.int32),
    )


def aggregate_annotations(
    image_id: str,
    ann_df: pd.DataFrame,
    orig_h: int,
    orig_w: int,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate multi-radiologist bounding boxes for a single image via WBF.

    Returns:
        boxes  : (N, 4) float32 normalized xyxy
        labels : (N,)   int32
    """
    rows = ann_df[ann_df["image_id"] == image_id]

    if rows.empty:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)

    # Group by radiologist so each rad contributes one "model"
    boxes_list, scores_list, labels_list = [], [], []
    for _, rad_rows in rows.groupby("rad_id"):
        b = rad_rows[["x_min", "y_min", "x_max", "y_max"]].values.astype(np.float32)
        # Normalize to [0, 1]
        b[:, [0, 2]] /= orig_w
        b[:, [1, 3]] /= orig_h
        b = np.clip(b, 0.0, 1.0)
        s = np.ones(len(b), dtype=np.float32)
        labels = rad_rows["class_id"].values.astype(np.float32)
        boxes_list.append(b.tolist())
        scores_list.append(s.tolist())
        labels_list.append(labels.tolist())

    fused_boxes, _, fused_labels = wbf_single_image(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr
    )
    return fused_boxes, fused_labels.astype(np.int32)


# ---------------------------------------------------------------------------
# YOLO label writer
# ---------------------------------------------------------------------------


def write_yolo_labels(
    image_id: str,
    boxes: np.ndarray,
    labels: np.ndarray,
    label_dir: str | Path,
) -> None:
    """
    Write a YOLO-format .txt label file.

    boxes  : (N, 4) normalized xyxy
    labels : (N,) int class indices
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_dir / f"{image_id}.txt"

    if len(boxes) == 0:
        out_path.write_text("")
        return

    lines = []
    for box, cls in zip(boxes, labels):
        # Map class_id to contiguous index (handles 14 -> 1 collapsing if LOCALIZE_ONLY)
        cls_idx = app_config.CLASS_ID_TO_IDX.get(int(cls), 0)

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_path.write_text("\n".join(lines))


def build_yolo_dataset(
    ann_df: pd.DataFrame,
    image_ids: list[str],
    img_dir: Path,
    label_dir: Path,
    iou_thr: float = 0.5,
    verbose: bool = True,
) -> None:
    """
    Pre-generate YOLO label .txt files for all images using multiple processes.
    Reads DICOM dims, runs WBF aggregation, writes labels.
    """
    import concurrent.futures
    import os
    from tqdm import tqdm

    label_dir.mkdir(parents=True, exist_ok=True)

    # Pre-filter ann_df to only the relevant image IDs to minimise per-worker work
    id_set = set(image_ids)
    sub_df = ann_df[ann_df["image_id"].isin(id_set)]

    # Build a plain-Python dict (fully picklable) keyed by image_id
    ann_lookup: dict[str, list] = {}
    for row in sub_df.itertuples(index=False):
        ann_lookup.setdefault(row.image_id, []).append(
            (row.class_id, row.rad_id, row.x_min, row.y_min, row.x_max, row.y_max)
        )

    def process_one(img_id: str) -> None:
        dicom_path = Path(img_dir) / f"{img_id}.dicom"
        if not dicom_path.exists():
            return
        orig_h, orig_w = get_image_dims(dicom_path)

        rows = ann_lookup.get(img_id, [])
        if not rows:
            write_yolo_labels(
                img_id,
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                label_dir,
            )
            return

        # Rebuild per-radiologist groups from the lookup dict
        from collections import defaultdict

        rad_groups: dict[str, list] = defaultdict(list)
        for cls_id, rad_id, x1, y1, x2, y2 in rows:
            rad_groups[rad_id].append((cls_id, x1, y1, x2, y2))

        boxes_list, scores_list, labels_list = [], [], []
        for rad_id, entries in rad_groups.items():
            b = np.array(
                [[x1, y1, x2, y2] for _, x1, y1, x2, y2 in entries], dtype=np.float32
            )
            b[:, [0, 2]] /= orig_w
            b[:, [1, 3]] /= orig_h
            b = np.clip(b, 0.0, 1.0)
            s = np.ones(len(b), dtype=np.float32)
            labels = np.array([cls_id for cls_id, *_ in entries], dtype=np.float32)
            boxes_list.append(b.tolist())
            scores_list.append(s.tolist())
            labels_list.append(labels.tolist())

        fused_boxes, _, fused_labels = wbf_single_image(
            boxes_list, scores_list, labels_list, iou_thr=iou_thr
        )
        write_yolo_labels(img_id, fused_boxes, fused_labels.astype(np.int32), label_dir)

    n_workers = os.cpu_count() or 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        if verbose:
            list(
                tqdm(
                    executor.map(process_one, image_ids),
                    total=len(image_ids),
                    desc="Writing YOLO labels",
                )
            )
        else:
            list(executor.map(process_one, image_ids))


def write_yolo_yaml(
    yaml_path: str | Path,
    train_img_dir: str | Path,
    val_img_dir: str | Path,
    class_names: list[str],
    test_img_dir: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> None:
    """Write a dataset YAML file for Ultralytics YOLO."""
    import yaml

    data = {}
    if dataset_path:
        data["path"] = str(dataset_path)

    data.update(
        {
            "train": str(train_img_dir),
            "val": str(val_img_dir),
            "nc": len(class_names),
            "names": class_names,
        }
    )
    if test_img_dir:
        data["test"] = str(test_img_dir)
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
