# Ensemble Performance Degradation Analysis and Fixes

## Overview
This document outlines the investigation into why the ensemble model using Weighted Box Fusion (WBF) was performing drastically worse (mAP: 0.119) than its underlying single components (YOLO: 0.231, Faster R-CNN: 0.165). Two separate root causes were identified and patched.

## Root Cause 1: Dynamic Weight Dropping (Bug)
### The Problem
In `scripts/stacking/ensemble.py`, the `_fuse_single` function contained a logic flaw where any detection arrays that were empty (i.e. model predicted 0 bounding boxes) were explicitly removed before passing the lists into the WBF algorithm.

```python
# Flawed logic:
non_empty = [
    (b, s, lbl, w)
    for b, s, lbl, w in zip(boxes_list, scores_list, labels_list, weights)
    if len(b) > 0 
]
```

Since the `ensemble_boxes` API normally normalizes the weights globally across models, dropping a model's empty predictions on a *per-image* basis essentially forced the WBF pipeline to rescale weights drastically based on whether objects were identified. Consequently, if YOLO found bounding box candidates but Faster R-CNN yielded none, Faster R-CNN's weight was decoupled, destroying the model scoring balance iteratively across the dataset.

### The Fix
Empty prediction arrays are no longer removed; instead, they are preserved and sent inside dummy arrays down to the multiclass fusion module where they map cleanly to 0 score weights without affecting the overall parameter index.

*Result:* mAP@50-95 restored from `0.119` to `0.215`. 

---

## Root Cause 2: Confidence Calibration Divergence
### The Problem
Even with the first bug eliminated, the ensemble slightly trailed the individual YOLO run (0.215 vs. 0.231). Upon sampling prediction subsets, a substantial discrepancy in raw confidence boundaries emerged:

*   **Faster R-CNN Score Bounds:** `[0.050, 0.999]`
*   **YOLO Score Bounds:** `[0.001, 0.847]`

Faster R-CNN operates on extremely inflated confidence margins, while YOLO retains realistically conservative constraints. WBF mathematically recalibrates fused coordinates relying directly on comparative confidence weights. Faster R-CNN's skewed distribution behaved as a "magnet", violently displacing YOLO's naturally accurate predictions toward False Positive centers.

### The Fix
To counteract this scale variance computationally, a new standard normalization flag has been injected into the inference pipeline via `--normalize-scores`. This uses a Min-Max scaling approach explicitly on each model’s output distribution to force outputs onto a smooth `{0.0...1.0}` frame globally prior to resolving fusion. 

Alternatively, if avoiding runtime compute overhead is desired, heavily skewing manual weights on YOLO (e.g. `--weights 0.75 0.25`) successfully buffers the disparity (bringing mAP@50-95 cleanly to `0.228`).
