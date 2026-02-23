# Running Guide

This guide explains how to use the executable scripts provided in the `scripts/` directory to run the full pipeline: from dataset preparation to model training, evaluation, and ensembling.

## 1. Preparation

Before training any neural network, you must convert the raw 16-bit DICOM images into optimized 3-channel 8-bit PNGs and generate YOLO-formatted bounding box labels.

```bash
# Basic usage (defaults to medium preset settings: 640x640)
python scripts/data/prepare.py

# Force rebuild at a specific resolution using multiple CPU cores
python scripts/data/prepare.py --image_size 1024 --workers 8 --force
```

This will create an `data/processed` folder containing the train, val, and test splits.

## 2. Training Models

Each architecture has its own dedicated training script. By default, training uses settings read from `paths.yaml` and `src/config.py`. 

You can easily control the GPU scaling and model size using the `--preset` flag (`small`, `medium`, or `large`).

### YOLOv8 (Phase 1: Localizer)
YOLO is used specifically as an abnormality localizer (all 14 medical classes are treated as bounding boxes). Since YOLO is fast, we recommend pushing the resolution as high as possible.
```bash
python scripts/yolo/train.py --preset large
```
*Note: YOLO handles dataset splitting implicitly via the `dataset.yaml` file generated in the preparation step.*

### Faster R-CNN
```bash
python scripts/faster_rcnn/train.py --preset medium --epochs 30
```

### DETR (Vision Transformer)
```bash
python scripts/detr/train.py --preset medium
```

### Tips for Training
* **Apple Silicon (Mac):** PyTorch natively supports the `mps` backend. If you run out of memory (OOM), add `--device cpu` to fall back to CPU memory (though training will be slower).
* **Previews:** Faster R-CNN and DETR will actively save JPEG previews of their bounding box predictions to `outputs/<model>/val_previews/` after every epoch. You can watch the models learn manually by opening these images!

## 3. Testing and Inference (JSONL Extraction)

After a model finishes training, you want to evaluate its best weights over the validation set. We use localized testing scripts to extract the predictions and dump them safely into a `.jsonl` logging format without consuming excessive RAM.

```bash
# Test YOLO and extract bounding boxes
python scripts/yolo/test.py \
    --weights outputs/yolo/train/weights/best.pt \
    --split val

# Test Faster R-CNN
python scripts/faster_rcnn/test.py \
    --weights outputs/faster_rcnn/best.pt \
    --split val

# Test DETR
python scripts/detr/test.py \
    --weights outputs/detr/best.pt \
    --split val
```

This will output `predictions_val.jsonl` files in their respective output directories.

## 4. Stacking & Ensembling

Once you have generated the JSONL prediction files from multiple different architectures (or multiple folds), you can fuse their bounding boxes together using Weighted Box Fusion (WBF). This greatly reduces false positives and tightens bounding boxes.

```bash
python scripts/ensemble/test.py \
    --preds outputs/yolo/predictions_val.jsonl outputs/faster_rcnn/predictions_val.jsonl outputs/detr/predictions_val.jsonl \
    --weights 2.0 1.5 1.0 \
    --ensemble_iou 0.5 \
    --ensemble_skip 0.05
```

### Parameters
* `--preds`: A space-separated list of the JSONL files you want to merge.
* `--weights`: A space-separated list of confidence weights for the models (in the same order as `--preds`). E.g., if YOLO tends to be highly accurate, give it a weight of `2.0`, while a weaker DETR model gets `1.0`.
* `--ensemble_iou`: How much two bounding boxes must overlap (IoU) before the WBF algorithm considers them to be the "same" object and merges them.

The fused output will be written to `outputs/ensemble/fused_predictions_wbf.jsonl`.
