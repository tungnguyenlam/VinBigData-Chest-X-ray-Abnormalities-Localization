import json
import os
import sys

# Ensure the paths resolve from project root
if os.getcwd().endswith('report'):
    os.chdir('../../')
sys.path.append(os.getcwd())

from docs.report.create_plot_utils.predictions import plot_prediction_example, plot_prediction_subplots

NUM_IMAGES = 10
OUTPUT_DIR = 'docs/report/images/prediction'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find images with predictions in the YOLO jsonl
image_ids = []
with open('outputs/yolo/yolo/predictions_test.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if len(data['boxes']) > 0:
            image_ids.append(data['image_id'])
            if len(image_ids) >= NUM_IMAGES:
                break

print(f"Generating prediction plots for {len(image_ids)} images...")

jsonl_paths = [
    'outputs/yolo/yolo/predictions_test.jsonl',
    'outputs/faster_rcnn/faster_rcnn/predictions_test.jsonl',
    'outputs/ensemble/test/fused_predictions_wbf.jsonl'
]

YOLO_THRESH = 0.4
FASTER_RCNN_THRESH = 0.4
ENSEMBLE_THRESH = 0.4

thresholds = [YOLO_THRESH, FASTER_RCNN_THRESH, ENSEMBLE_THRESH]

for img_id in image_ids:
    image_path = f"data/processed/test/images/{img_id}.png"
    titles = [f"YOLO v8: {img_id}", f"Faster R-CNN: {img_id}", f"Ensemble WBF: {img_id}"]
    
    plot_prediction_subplots(
        image_path=image_path,
        jsonl_paths=jsonl_paths,
        titles=titles,
        output_path=os.path.join(OUTPUT_DIR, f'{img_id}_comparison.png'),
        labels_dir='data/processed/test/labels',
        thresholds=thresholds
    )
    
    print(f"Generated plots for image: {img_id}")

print(f"Finished generating all plots in {OUTPUT_DIR}")
