#!/usr/bin/env python3
"""
Create Inference Examples with Thumbnail Display
This script creates inference examples showing both bounding boxes and segmentation masks
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def create_thumbnail_grid(image_dir, output_path, max_images=4, figsize=(15, 10)):
    """Create a thumbnail grid showing inference results"""
    
    # Get image files
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(list(Path(image_dir).glob(ext)))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    # Limit to max_images
    image_files = image_files[:max_images]
    
    # Calculate grid size
    n_images = len(image_files)
    if n_images <= 2:
        rows, cols = 1, n_images
    elif n_images <= 4:
        rows, cols = 2, 2
    else:
        rows = int(np.ceil(n_images / 2))
        cols = 2
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if n_images > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot images
    for i, img_path in enumerate(image_files):
        if i >= len(axes):
            break
            
        # Read and display image
        img = cv2.imread(str(img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].set_title(f"{img_path.name}", fontsize=10)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Error loading\n{img_path.name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Thumbnail grid saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create inference thumbnail examples')
    parser.add_argument('--inference-dir', type=str, 
                       default='d:/241-353/runs/predict-seg/food_inference_demo',
                       help='Directory containing inference results')
    parser.add_argument('--output-dir', type=str,
                       default='d:/241-353/inference_thumbnails',
                       help='Output directory for thumbnails')
    parser.add_argument('--max-images', type=int, default=4,
                       help='Maximum number of images to include in thumbnail')
    
    args = parser.parse_args()
    
    inference_dir = Path(args.inference_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating Inference Thumbnail Examples")
    print(f"Inference directory: {inference_dir}")
    print(f"Output directory: {output_dir}")
    
    if not inference_dir.exists():
        print(f"Error: Inference directory {inference_dir} does not exist!")
        print("Please run inference first using:")
        print("python train_and_inference_demo.py")
        return
    
    # Create validation set thumbnails
    val_thumbnail_path = output_dir / "validation_inference_examples.png"
    create_thumbnail_grid(inference_dir, val_thumbnail_path, args.max_images)
    
    # Create training set thumbnails if available
    train_inference_dir = inference_dir.parent / "food_inference_train"
    if train_inference_dir.exists():
        train_thumbnail_path = output_dir / "training_inference_examples.png"
        create_thumbnail_grid(train_inference_dir, train_thumbnail_path, args.max_images)
    
    # Define summary_path at the beginning
    summary_path = output_dir / "inference_summary.txt"

    # Create test set thumbnails if available
    test_inference_dir = inference_dir.parent / "food_inference_test"
    if test_inference_dir.exists():
        test_thumbnail_path = output_dir / "test_inference_examples.png"
        create_thumbnail_grid(test_inference_dir, test_thumbnail_path, args.max_images)

        # Add test thumbnails to the summary report
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write(f"- test_inference_examples.png\n")

    # Create a summary report
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("YOLOv5 Segmentation Inference Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Inference Results Directory: {inference_dir}\n")
        f.write(f"Output Directory: {output_dir}\n\n")

        # Count files
        image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            image_files.extend(list(inference_dir.glob(ext)))

        f.write(f"Total inference images: {len(image_files)}\n")

        # List label files if available
        label_files = list(inference_dir.glob('labels/*.txt'))
        f.write(f"Total label files: {len(label_files)}\n\n")

        f.write("Generated Thumbnails:\n")
        f.write(f"- {val_thumbnail_path.name}\n")
        if train_inference_dir.exists():
            f.write(f"- training_inference_examples.png\n")
        if test_inference_dir.exists():
            f.write(f"- test_inference_examples.png\n")

        f.write("\nFeatures shown in inference results:\n")
        f.write("✅ Detected bounding boxes\n")
        f.write("✅ Segmentation masks (overlay)\n")
        f.write("✅ Class labels with confidence scores\n")
        f.write("✅ Color-coded instances\n")
    
    print(f"\nSummary report saved to: {summary_path}")
    print("\nInference thumbnail examples created successfully!")
    print(f"Check the output directory: {output_dir}")

if __name__ == "__main__":
    main()
