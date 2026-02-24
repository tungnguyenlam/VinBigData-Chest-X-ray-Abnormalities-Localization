# FROC Evaluation Strategy

## Overview

We use **FROC (Free-Response Receiver Operating Characteristic)** as the primary evaluation metric alongside COCO-style mAP. FROC is the standard benchmark in medical imaging for lesion detection tasks and is more clinically meaningful than traditional metrics for our VinBigData Chest X-ray Abnormalities Detection problem.

## Why FROC Over mAP / Precision / Recall / F1

### The Problem with Traditional Metrics

| Issue | Precision/Recall/F1 | FROC |
|-------|---------------------|------|
| Multiple lesions per image | Only handles binary (has disease / no disease) at image level | Evaluates each lesion independently with spatial localization |
| Localization accuracy | Cannot distinguish "found disease but wrong location" from correct detection | Requires IoU overlap with ground truth box to count as TP |
| Unlimited predictions per image | Penalizes extra detections uniformly | Uses FP/image rate — directly tells you "how many false alarms per scan" |
| Varying number of GT lesions | Unknown denominator distorts recall | Counts every GT lesion explicitly (Lesion Localization Fraction) |
| Clinical interpretability | Abstract numbers | "At 1 false positive per image, we detect 85% of lesions" — directly actionable |

### Why It Fits Our Approach

1. **Lesion-level evaluation**: We use `LOCALIZE_ONLY = True`, collapsing all 14 pathology classes into a single "Abnormality" class. FROC evaluates at the lesion level — *exactly* the granularity we optimize for.

2. **Multi-lesion images**: The VinBigData dataset has images with 0 to many lesions. FROC handles this naturally — each lesion is scored independently.

3. **No-finding images as negatives**: We include "No finding" images as hard negatives during training. FROC's X-axis (FP/image) directly measures how well the model avoids false alarms on these images.

4. **Multi-annotator ground truth**: VinBigData uses 3 radiologists. After WBF deduplication during preprocessing, FROC's greedy matching (each GT box matched at most once) is well-suited to the deduplicated annotations.

## How FROC Works

### Definition

The FROC curve plots:
- **Y-axis**: Lesion Localization Fraction (LLF) — the fraction of ground truth lesions correctly detected
- **X-axis**: Average number of false positives per image (FP/image)

$$LLF = \frac{\text{Number of correctly localized lesions}}{\text{Total ground truth lesions}}$$

$$\text{FP/image} = \frac{\text{Total false positive detections}}{\text{Total number of images}}$$

### FROC Score

The **FROC score** is the mean sensitivity at 7 standard FP/image operating points:

```
FP/image rates: [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
FROC score = mean(sensitivity at each rate)
```

This gives a single scalar that summarizes model performance across a clinically relevant range of false positive trade-offs.

### Matching Algorithm

For each image:

1. All predictions are sorted by confidence score (descending).
2. For each prediction, compute IoU with all unmatched GT boxes.
3. If the best IoU ≥ threshold (default 0.5) and that GT box hasn't been matched yet:
   - Mark prediction as **True Positive** (TP)
   - Mark GT box as matched (cannot be matched again)
4. Otherwise: mark prediction as **False Positive** (FP)
5. GT boxes left unmatched are **False Negatives** (FN)

This is a **greedy matching** strategy — predictions are matched in order of decreasing confidence, and each GT box can only be claimed once.

### Building the FROC Curve

1. Collect all (score, is_TP) pairs across the entire dataset.
2. Sort by descending score.
3. Sweep through: at each threshold, compute cumulative TP and FP counts.
4. Convert to `sensitivity = cum_TP / total_GT_lesions` and `FP/image = cum_FP / total_images`.
5. Interpolate sensitivity at each standard FP/image rate.

## Implementation

### Core Function

Located in `scripts/predict_logger.py`:

```python
from scripts.predict_logger import evaluate_froc

froc_result = evaluate_froc(
    pred_file="outputs/yolo/predictions_val.jsonl",
    data_cfg=data_cfg,
    split="val",
    iou_threshold=0.5,       # IoU threshold for TP matching
    fp_rates=(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),  # standard operating points
    prepared_dataset_root="data/processed",
)
```

Returns:
```python
{
    "froc_score": 0.7234,           # mean sensitivity across operating points
    "sensitivities": [0.45, 0.58, 0.68, 0.78, 0.85, 0.89, 0.93],
    "fp_rates": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "total_gt_lesions": 2438,
    "total_images": 750,
    "iou_threshold": 0.5,
}
```

### Test Script Usage

FROC is computed automatically in all test scripts — no extra flags needed:

```bash
# YOLO
python scripts/yolo/test.py --weights outputs/yolo/best.pt --split val

# Faster R-CNN
python scripts/faster_rcnn/test.py --weights outputs/faster_rcnn/best.pt --split val

# DETR
python scripts/detr/test.py --weights outputs/detr/best.pt --split val
```

Each test script prints:
```
  === Results ===
  mAP@50-95: 0.2541
  mAP@50: 0.4312
  mAP@75: 0.2103

  === FROC Results (IoU=0.5) ===
  FROC Score: 0.7234
  GT Lesions: 2438  |  Images: 750
    Sensitivity @ 0.125 FP/img: 0.4500
    Sensitivity @ 0.25 FP/img:  0.5800
    Sensitivity @ 0.5 FP/img:   0.6800
    Sensitivity @ 1.0 FP/img:   0.7800
    Sensitivity @ 2.0 FP/img:   0.8500
    Sensitivity @ 4.0 FP/img:   0.8900
    Sensitivity @ 8.0 FP/img:   0.9300
```

All FROC metrics are saved alongside mAP in the `*.metrics.json` output file.

### Dependencies

**No additional dependencies.** The FROC computation uses only NumPy, which is already a project dependency. No torchmetrics required for FROC (unlike mAP).

## Interpreting Results

| FROC Score | Interpretation |
|-----------|----------------|
| 0.0 – 0.3 | Poor — Model misses most lesions or produces excessive false positives |
| 0.3 – 0.5 | Fair — Some lesions detected, but significant misses at low FP rates |
| 0.5 – 0.7 | Good — Reasonable detection with acceptable false positive trade-off |
| 0.7 – 0.85 | Very good — Strong detection across most operating points |
| 0.85 – 1.0 | Excellent — High sensitivity even at very low FP rates |

### What to Optimize For

- **Clinical deployment**: Focus on sensitivity at low FP rates (0.125–1.0 FP/img). Radiologists tolerate few false alarms.
- **Screening use case**: Sensitivity at 2.0–4.0 FP/img is acceptable — missing a lesion is worse than an extra flag.
- **Overall ranking**: The FROC score (mean across all 7 rates) gives a balanced single number.

## References

- Bandos, A.I., et al. "Free-Response Methodology: Alternate Analysis and a New Observer-Performance Experiment." *Radiology*, 2009.
- Chakraborty, D.P. "A Brief History of FROC Paradigm Data Analysis." *Academic Radiology*, 2013.
- VinBigData Chest X-ray Abnormalities Detection — [Kaggle Competition](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
