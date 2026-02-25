import cv2
import json
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

def plot_prediction_example(image_path, jsonl_path, output_path, title="Prediction Example", threshold=0.0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    boxes = []
    scores = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['image_id'] == image_id:
                boxes = data.get('boxes', [])
                scores = data.get('scores', [])
                break
                
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_rgb)
    
    for idx, box_info in enumerate(boxes):
        score = scores[idx] if idx < len(scores) else 1.0
        
        if score < threshold:
            continue
        
        if len(box_info) >= 4:
            x_min, y_min, x_max, y_max = box_info[:4]
            
            if x_max <= 2.0 and y_max <= 2.0:
                x_min, x_max = x_min * w, x_max * w
                y_min, y_max = y_min * h, y_max * h
                
            width_box, height_box = x_max - x_min, y_max - y_min
            
            rect = patches.Rectangle((x_min, y_min), width_box, height_box, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            ax.text(x_min, y_min - 5, f"Anomaly: {score:.2f}", color='red', fontsize=12, backgroundcolor='white')
            
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

import pandas as pd

def _draw_boxes_on_ax(ax, img_rgb, image_id, jsonl_path, h, w, gt_boxes=None, threshold=0.0):
    if gt_boxes is None:
        gt_boxes = []
        
    boxes = []
    scores = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data['image_id'] == image_id:
                boxes = data.get('boxes', [])
                scores = data.get('scores', [])
                break
                
    ax.imshow(img_rgb)
    
    # Draw GT Boxes (Green, dashed)
    for box in gt_boxes:
        x_min, y_min, x_max, y_max = box[:4]
        if pd.isna(x_min): continue
        width_box, height_box = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width_box, height_box, linewidth=2, edgecolor='green', facecolor='none', linestyle='dashed')
        ax.add_patch(rect)
        if len(box) > 4:
            ax.text(x_min, y_max + 15, str(box[4]), color='green', fontsize=10, backgroundcolor='white', weight='bold')

    # Draw Pred Boxes (Red, solid)
    for idx, box_info in enumerate(boxes):
        score = scores[idx] if idx < len(scores) else 1.0
        
        if score < threshold:
            continue
            
        if len(box_info) >= 4:
            x_min, y_min, x_max, y_max = box_info[:4]
            
            if x_max <= 2.0 and y_max <= 2.0:
                x_min, x_max = x_min * w, x_max * w
                y_min, y_max = y_min * h, y_max * h
                
            width_box, height_box = x_max - x_min, y_max - y_min
            
            rect = patches.Rectangle((x_min, y_min), width_box, height_box, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, f"Anomaly: {score:.2f}", color='red', fontsize=12, backgroundcolor='white')

    ax.axis('off')


# We are doing anomaly localization only, so we use a single class name
MODELS_CLASS_NAME = 'Anomaly'

def plot_prediction_subplots(image_path, jsonl_paths, titles, output_path, labels_dir=None, thresholds=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if thresholds is None:
        thresholds = [0.0] * len(jsonl_paths)
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    gt_boxes = []
    if labels_dir and os.path.exists(labels_dir):
        label_file = os.path.join(labels_dir, f"{image_id}.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # cx, cy, bw, bh are normalized [0, 1]
                        cx, cy, bw, bh = map(float, parts[1:5])
                        
                        x_min = (cx - bw / 2) * w
                        y_min = (cy - bh / 2) * h
                        x_max = (cx + bw / 2) * w
                        y_max = (cy + bh / 2) * h
                        
                        if class_id == 14: # Skip 'No finding'
                            continue
                            
                        class_name = MODELS_CLASS_NAME
                        gt_boxes.append([x_min, y_min, x_max, y_max, class_name])
            
    fig, axes = plt.subplots(1, len(jsonl_paths), figsize=(20, 8))
    if len(jsonl_paths) == 1:
        axes = [axes]
        
    for ax, jsonl_path, title, thresh in zip(axes, jsonl_paths, titles, thresholds):
        _draw_boxes_on_ax(ax, img_rgb, image_id, jsonl_path, h, w, gt_boxes=gt_boxes, threshold=thresh)
        ax.set_title(title, fontsize=16)
        
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
