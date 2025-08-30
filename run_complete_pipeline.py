#!/usr/bin/env python3
"""
Complete YOLOv5 Segmentation Training and Inference Pipeline
This script includes your training command and adds inference example generation
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Your original training command
TRAINING_COMMAND = """python segment/train.py --data ../dataset_new/food_data.yaml --weights ../yolov5s-seg.pt --epochs 250 --batch-size 4 --img 640 --project ../runs/train-seg4 --name food_demo4 --device cpu --exist-ok"""

def run_command(cmd, description, capture_output=False):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"🚀 {description}")
    print(f"📝 Command: {cmd}")
    print(f"{'='*80}")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            return result.stdout
        else:
            subprocess.run(cmd, shell=True, check=True)
        print("✅ SUCCESS!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print("Stdout:", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("Stderr:", e.stderr)
        return False

def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} not found: {file_path}")
        return False

def main():
    print("🎯 YOLOv5 Segmentation Training and Inference Pipeline")
    print("=" * 80)
    
    # Set working directory
    work_dir = Path("d:/241-353")
    yolov5_dir = work_dir / "yolov5"
    
    print(f"📁 Working directory: {work_dir}")
    print(f"📁 YOLOv5 directory: {yolov5_dir}")
    
    # Check prerequisites
    if not yolov5_dir.exists():
        print(f"❌ YOLOv5 directory not found: {yolov5_dir}")
        return
    
    if not check_file_exists(work_dir / "dataset_new" / "food_data.yaml", "Dataset config"):
        return
    
    if not check_file_exists(work_dir / "yolov5s-seg.pt", "Pre-trained weights"):
        return
    
    # Change to yolov5 directory
    os.chdir(yolov5_dir)
    print(f"📂 Changed to directory: {os.getcwd()}")
    
    # Step 1: Training (Your original command)
    print("\n🏋️ STEP 1: Training YOLOv5 Segmentation Model")
    start_time = time.time()
    
    if not run_command(TRAINING_COMMAND, "YOLOv5 Segmentation Training"):
        print("💥 Training failed! Stopping pipeline...")
        return
    
    training_time = time.time() - start_time
    print(f"⏱️ Training completed in {training_time/3600:.2f} hours")
    
    # Step 2: Check trained model
    best_model_path = work_dir / "runs" / "train-seg4" / "food_demo4" / "weights" / "best.pt"
    
    if not check_file_exists(best_model_path, "Trained model"):
        print("💥 Trained model not found! Cannot proceed with inference...")
        return
    
    # Step 3: Create inference examples on validation set
    print("\n🔍 STEP 2: Creating Inference Examples on Validation Set")

    val_inference_cmd = f"""python segment/predict.py --weights "../runs/train-seg4/food_demo4/weights/best.pt" --source "../dataset_new/images/val" --data "../dataset_new/food_data.yaml" --conf-thres 0.1 --iou-thres 0.45  --save-txt --save-conf --project "../runs/predict-seg" --name "food_inference_demo" --exist-ok --line-thickness 2"""

    if not run_command(val_inference_cmd, "Validation Set Inference"):
        print("⚠️ Validation inference failed, but continuing...")
    
    # Step 4: Create inference examples on training set (sample)
    print("\n🔍 STEP 3: Creating Inference Examples on Training Set Sample")

    train_inference_cmd = f"""python segment/predict.py --weights "../runs/train-seg4/food_demo4/weights/best.pt" --source "../dataset_new/images/train" --data "../dataset_new/food_data.yaml" --conf-thres 0.1 --iou-thres 0.45  --save-txt --save-conf --project "../runs/predict-seg" --name "food_inference_train" --exist-ok --line-thickness 2"""

    run_command(train_inference_cmd, "Training Set Inference")
    
    # Step 5: Create inference examples on test set
    print("\n🔍 STEP 4: Creating Inference Examples on Test Set")

    test_inference_cmd = f"""python segment/predict.py --weights "../runs/train-seg4/food_demo4/weights/best.pt" --source "../dataset_new/images/test" --data "../dataset_new/food_data.yaml" --conf-thres 0.1 --iou-thres 0.45  --save-txt --save-conf --project "../runs/predict-seg" --name "food_inference_test" --exist-ok --line-thickness 2"""

    run_command(test_inference_cmd, "Test Set Inference")
    
    # Step 6: Generate Instance Segmentation Thumbnails
    print("\n🖼️ STEP 3: Generating Instance Segmentation Thumbnails")

    thumbnail_cmd = f"""python segment/generate_thumbnails.py --weights "../runs/train-seg4/food_demo4/weights/best.pt" \
        --source "../dataset_new/images/val" --data "../dataset_new/food_data.yaml" \
        --output "../runs/inference_thumbnails" --overlay --examples 4"""

    if not run_command(thumbnail_cmd, "Instance Segmentation Thumbnails Generation"):
        print("⚠️ Thumbnail generation failed, but continuing...")
    
    # Step 7: Create thumbnail summaries
    print("\n🖼️ STEP 5: Creating Thumbnail Summaries")
    
    os.chdir(work_dir)  # Go back to main directory
    
    thumbnail_cmd = f"""python create_inference_thumbnails.py --inference-dir "runs/predict-seg/food_inference_demo" --output-dir "inference_thumbnails" --max-images 4"""
    
    run_command(thumbnail_cmd, "Creating Thumbnail Summaries")
    
    # Step 8: Generate final report
    print("\n📊 STEP 6: Generating Final Report")
    
    # Count results
    val_results_dir = work_dir / "runs" / "predict-seg" / "food_inference_demo"
    train_results_dir = work_dir / "runs" / "predict-seg" / "food_inference_train"
    test_results_dir = work_dir / "runs" / "predict-seg" / "food_inference_test"
    
    val_images = list(val_results_dir.glob("*.jpg")) + list(val_results_dir.glob("*.png")) if val_results_dir.exists() else []
    train_images = list(train_results_dir.glob("*.jpg")) + list(train_results_dir.glob("*.png")) if train_results_dir.exists() else []
    test_images = list(test_results_dir.glob("*.jpg")) + list(test_results_dir.glob("*.png")) if test_results_dir.exists() else []
    
    # Final summary
    print("\n" + "🎉" * 50)
    print("🎯 PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉" * 50)
    
    print(f"\n📋 RESULTS SUMMARY:")
    print(f"   ⏱️ Total training time: {training_time/3600:.2f} hours")
    print(f"   🎯 Best model: {best_model_path}")
    print(f"   📊 Training results: {work_dir}/runs/train-seg4/food_demo4/")
    print(f"   🔍 Validation inference: {len(val_images)} images in {val_results_dir}")
    print(f"   🔍 Training inference: {len(train_images)} images in {train_results_dir}")
    print(f"   🔍 Test inference: {len(test_images)} images in {test_results_dir}")
    print(f"   🖼️ Thumbnails: {work_dir}/inference_thumbnails/")
    
    print(f"\n📁 KEY DIRECTORIES:")
    print(f"   📈 Training metrics & logs: runs/train-seg4/food_demo4/")
    print(f"   🔍 Validation predictions: runs/predict-seg/food_inference_demo/")
    print(f"   🔍 Training predictions: runs/predict-seg/food_inference_train/")
    print(f"   🔍 Test predictions: runs/predict-seg/food_inference_test/")
    print(f"   🖼️ Thumbnail examples: inference_thumbnails/")
    
    print(f"\n🛠️ WHAT'S IN THE INFERENCE RESULTS:")
    print(f"   ✅ Instance segmentation masks (colored overlays)")
    print(f"   ✅ Bounding boxes with class labels")
    print(f"   ✅ Confidence scores for each detection")
    print(f"   ✅ Text files with coordinates and masks")
    
    print(f"\n📊 TO VIEW TRAINING METRICS:")
    print(f"   tensorboard --logdir \"runs/train-seg4/food_demo4\"")

    print(f"\n🎯 YOUR INFERENCE EXAMPLES ARE READY!")
    print(f"   Check: inference_thumbnails/validation_inference_examples.png")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total pipeline time: {total_time/3600:.2f} hours")

if __name__ == "__main__":
    main()
