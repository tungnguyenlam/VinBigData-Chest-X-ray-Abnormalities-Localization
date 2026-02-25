import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_finding_distribution(image_dir, csv_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    image_files = glob.glob(os.path.join(image_dir, '*.*'))
    total_images = len(image_files)
    
    if total_images == 0:
        print("No images found in", image_dir)
        return
        
    df = pd.read_csv(csv_path)
    images_with_finding = df['image_id'].nunique()
    images_no_finding = total_images - images_with_finding
    
    labels = ['Have finding', 'No finding']
    sizes = [images_with_finding, images_no_finding]
    colors = ['#ff9999','#66b3ff']
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
    plt.title('Distribution of Images with and without Findings', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_box_count_distribution(csv_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(csv_path)
    box_counts = df.groupby('image_id').size()
    
    plt.figure(figsize=(10, 6))
    plt.hist(box_counts, bins=range(1, box_counts.max() + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
    plt.xlabel('Number of Bounding Boxes', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.title('Distribution of Bounding Box Counts per Image', fontsize=16)
    plt.xticks(range(1, box_counts.max() + 1))
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
