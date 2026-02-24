# Model Stacking (Ensembling)

Model stacking (or ensembling) in object detection is the process of combining the bounding box predictions from multiple independently trained models to produce a single, highly accurate set of predictions.

For the VinBigData Chest X-ray dataset, we use an ensemble of **YOLO**, **Faster R-CNN**, and **DETR**. Since these models have fundamentally different architectures (anchor-free, anchor-based, and transformer-based), they learn diverse representations and make different types of errors. Ensembling them significantly boosts overall precision and recall.

---

## 🚀 The Pipeline

The ensemble pipeline operates on **prediction files** rather than holding all models in GPU memory simultaneously. This allows you to ensemble large models regardless of your available VRAM.

### Step 1: Predict on Validation Split
Run each model's test script on the validation split separately:
```bash
python scripts/yolo/test.py --weights outputs/yolo/best.pt --split val
python scripts/faster_rcnn/test.py --weights outputs/faster_rcnn/best.pt --split val
python scripts/detr/test.py --weights outputs/detr/best.pt --split val
```
*Outputs:* `outputs/{model}/predictions_val.jsonl`

### Step 2: Tune Ensemble Weights
Because some models perform better than others, we weight their predictions. Use the tuning script to find the optimal combination on the validation set:
```bash
python scripts/stacking/tune_weights.py \
  --yolo-pred outputs/yolo/predictions_val.jsonl \
  --frcnn-pred outputs/faster_rcnn/predictions_val.jsonl \
  --detr-pred outputs/detr/predictions_val.jsonl
```
*Outputs:* `outputs/ensemble/val/tuned_weights.json`

*(Note: Adding `--grid-search` sweeps all combinations to find the absolute best val mAP, but takes much longer than the default pro-rata approach).*

### Step 3: Predict on Test Split
Generate test split predictions for each model:
```bash
python scripts/yolo/test.py --weights outputs/yolo/best.pt --split test
python scripts/faster_rcnn/test.py --weights outputs/faster_rcnn/best.pt --split test
python scripts/detr/test.py --weights outputs/detr/best.pt --split test
```

### Step 4: Fuse with Tuned Weights
Combine the test predictions using the weights learned in Step 2:
```bash
python scripts/stacking/test.py \
  --yolo-pred outputs/yolo/predictions_test.jsonl \
  --frcnn-pred outputs/faster_rcnn/predictions_test.jsonl \
  --detr-pred outputs/detr/predictions_test.jsonl \
  --split test \
  --tuned-weights outputs/ensemble/val/tuned_weights.json
```
*Outputs:* `outputs/ensemble/test/fused_predictions_wbf.jsonl`

---

## 🧠 Fusion Methods

When three models predict bounding boxes for an "Aortic enlargement", you might end up with three slightly overlapping boxes. We need an algorithm to merge them into one.

### 1. Weighted Box Fusion (WBF) — ⭐ Preferred Method
Instead of simply dropping boxes, WBF uses **all** bounding boxes to construct the final box. 

**Mathematical Formulation:**
Let there be $T$ models in the ensemble, each assigned a structural weight $W_m$ (where $m \in \{1, \dots, T\}$).
1. **Score Adjustment**: For each predicted bounding box with confidence score $C$ from model $m$, its adjusted score becomes:
   $$S = C \times W_m$$
2. **Clustering**: All boxes from all models are pooled and sorted in descending order of $S$. They are then grouped into clusters based on an IoU (Intersection over Union) threshold. If a box overlaps an existing cluster with IoU > threshold, it joins the cluster.
3. **Box Fusion**: For a cluster containing $N$ boxes, the final fused coordinates ($X_{1}, Y_{1}, X_{2}, Y_{2}$) are computed as a weighted average. The coordinates of each box $i$ ($x_{1_i}, y_{1_i}, x_{2_i}, y_{2_i}$) are weighted by their adjusted scores $S_i$:
   $$X_{1, \text{fused}} = \frac{\sum_{i=1}^{N} S_i \cdot x_{1_i}}{\sum_{i=1}^{N} S_i}$$
   *(The same weighted average formula applies to $Y_1, X_2$, and $Y_2$)*.
4. **Score Fusion**: The confidence score of the final fused box is calculated to reward instances where multiple models agreed. It is typically the sum of the scores divided by the total number of models $T$:
   $$S_{\text{fused}} = \frac{\sum_{i=1}^{N} S_i}{T}$$
   *(This implicitly penalizes predictions made by only one model, as $N < T$ means the sum is divided by a larger number).*

* **Why it's best for medical imaging**: X-ray abnormalities often have fuzzy, indistinct boundaries. Averaging the coordinates from multiple models produces a more stable and accurate bounding box than picking just one model's box.

### 2. Non-Maximum Suppression (NMS)
The traditional approach. It picks the single box with the highest confidence score and deletes any overlapping boxes (above the IoU threshold).
* **Drawback**: Discards information; if a lower-confidence model had a better-fitting box, that box is permanently lost.

### 3. Soft-NMS
Similar to NMS, but instead of deleting overlapping boxes outright, it heavily decays their confidence scores. 

## 🛠 Directory Structure

Outputs are strictly segregated by split to prevent data leakage:

```text
outputs/
├── yolo/
│   ├── predictions_val.jsonl
│   └── predictions_test.jsonl
├── faster_rcnn/
│   ├── predictions_val.jsonl
│   └── predictions_test.jsonl
├── detr/
│   ├── predictions_val.jsonl
│   └── predictions_test.jsonl
└── ensemble/
    ├── val/
    │   ├── tuned_weights.json                    <-- Learned from val split
    │   ├── fused_predictions_wbf.jsonl
    │   └── fused_predictions_wbf.metrics.json
    └── test/
        ├── fused_predictions_wbf.jsonl
        └── fused_predictions_wbf.metrics.json
```
