import cv2
import matplotlib.pyplot as plt
import os

def visualize_channels(image_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    channel_names = ['Channel 0 (Original)', 'Channel 1 (CLAHE)', 'Channel 2 (Laplacian)']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title('Combined RGB (Advanced Preprocessing)', fontsize=14)
    axes[0].axis('off')
    
    for i in range(3):
        axes[i+1].imshow(img_rgb[:, :, i], cmap='gray')
        axes[i+1].set_title(channel_names[i], fontsize=14)
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
