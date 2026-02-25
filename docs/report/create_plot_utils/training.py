import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_yolo_loss_history(csv_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    epochs = df['epoch']
    
    train_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    val_loss = df['val/box_loss'] + df['val/cls_loss'] + df['val/dfl_loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss (Box+Cls+Dfl)', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss (Box+Cls+Dfl)', color='orange', linewidth=2)
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Total Loss', fontsize=14)
    plt.title('YOLO Training and Validation Loss History', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_faster_rcnn_loss_history(csv_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    train_loss = df['train_loss']
    val_loss = df['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)
    
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Faster R-CNN Training and Validation Loss History', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
