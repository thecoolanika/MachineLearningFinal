"""
YOLOv8 object detection script for fire localization.

This script uses a pre-trained YOLOv8 model to detect and localize fires
in images from the FlameVision dataset. It focuses on running inference
and understanding the outputs rather than full-scale training.
"""

import sys
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found. Please install it with: pip install ultralytics")
    sys.exit(1)


def load_sample_images(data_dir: str, split: str = 'test', max_images: int = 50):
    """
    Load a sample of images from the dataset for YOLOv8 inference.
    
    Args:
        data_dir: Root directory of the dataset
        split: Dataset split to use ('train', 'val', or 'test')
        max_images: Maximum number of images to process
        
    Returns:
        List of image paths
    """
    data_path = Path(data_dir) / split
    
    image_paths = []
    
    # Load fire images
    fire_dir = data_path / 'fire'
    if fire_dir.exists():
        fire_images = list(fire_dir.glob('*.png')) + list(fire_dir.glob('*.jpg'))
        image_paths.extend(fire_images[:max_images // 2])
    
    # Load non-fire images
    non_fire_dir = data_path / 'non_fire'
    if non_fire_dir.exists():
        non_fire_images = list(non_fire_dir.glob('*.png')) + list(non_fire_dir.glob('*.jpg'))
        image_paths.extend(non_fire_images[:max_images // 2])
    
    return image_paths[:max_images]


def run_yolov8_inference(model, image_paths, conf_threshold=0.25, save_results=True, output_dir='results/yolov8'):
    """
    Run YOLOv8 inference on a set of images.
    
    Args:
        model: YOLOv8 model
        image_paths: List of image paths
        conf_threshold: Confidence threshold for detections
        save_results: Whether to save visualization results
        output_dir: Directory to save results
        
    Returns:
        Dictionary with inference statistics
    """
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    inference_times = []
    detection_counts = []
    total_detections = 0
    images_with_detections = 0
    
    print(f"\nRunning inference on {len(image_paths)} images...")
    print("=" * 60)
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        # Run inference
        start_time = time.time()
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Process results
        result = results[0]
        num_detections = len(result.boxes)
        detection_counts.append(num_detections)
        total_detections += num_detections
        
        if num_detections > 0:
            images_with_detections += 1
        
        # Save visualization if requested
        if save_results and num_detections > 0:
            # Get annotated image
            annotated_img = result.plot()
            output_file = output_path / f"{img_path.stem}_detections.jpg"
            cv2.imwrite(str(output_file), annotated_img)
    
    # Calculate statistics
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    min_inference_time = np.min(inference_times)
    max_inference_time = np.max(inference_times)
    avg_detections = np.mean(detection_counts)
    
    stats = {
        'num_images': len(image_paths),
        'total_detections': total_detections,
        'images_with_detections': images_with_detections,
        'avg_detections_per_image': avg_detections,
        'avg_inference_time': avg_inference_time,
        'std_inference_time': std_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'total_time': sum(inference_times),
        'fps': len(image_paths) / sum(inference_times) if sum(inference_times) > 0 else 0
    }
    
    return stats


def main():
    """Main function for YOLOv8 detection."""
    parser = argparse.ArgumentParser(description='Run YOLOv8 fire detection')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory of the dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum number of images to process')
    parser.add_argument('--model_name', type=str, default='yolov8n.pt',
                       help='YOLOv8 model name (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save visualization results')
    parser.add_argument('--output_dir', type=str, default='results/yolov8',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {args.model_name}")
    print("Note: This will download the model if not already present.")
    try:
        model = YOLO(args.model_name)
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        print("Trying to download from ultralytics...")
        model = YOLO(args.model_name)
    
    print(f"Model loaded successfully!")
    print(f"Model classes: {model.names}")
    
    # Load sample images
    print(f"\nLoading images from {args.split} split...")
    image_paths = load_sample_images(args.data_dir, args.split, args.max_images)
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {args.data_dir}/{args.split}")
        return
    
    print(f"Loaded {len(image_paths)} images")
    
    # Run inference
    stats = run_yolov8_inference(
        model,
        image_paths,
        conf_threshold=args.conf_threshold,
        save_results=args.save_results,
        output_dir=str(output_dir)
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("YOLOv8 Inference Results")
    print("=" * 60)
    print(f"Number of images processed: {stats['num_images']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Images with detections: {stats['images_with_detections']} ({100 * stats['images_with_detections'] / stats['num_images']:.1f}%)")
    print(f"Average detections per image: {stats['avg_detections_per_image']:.2f}")
    print(f"\nInference Time Statistics:")
    print(f"  Average: {stats['avg_inference_time']*1000:.2f} ms")
    print(f"  Std Dev: {stats['std_inference_time']*1000:.2f} ms")
    print(f"  Min: {stats['min_inference_time']*1000:.2f} ms")
    print(f"  Max: {stats['max_inference_time']*1000:.2f} ms")
    print(f"  Total: {stats['total_time']:.2f} s")
    print(f"  FPS: {stats['fps']:.2f}")
    
    # Save results to file
    results_file = output_dir / 'yolov8_inference_results.txt'
    with open(results_file, 'w') as f:
        f.write("YOLOv8 Fire Detection - Inference Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset split: {args.split}\n")
        f.write(f"Confidence threshold: {args.conf_threshold}\n")
        f.write(f"\nStatistics:\n")
        f.write(f"  Number of images processed: {stats['num_images']}\n")
        f.write(f"  Total detections: {stats['total_detections']}\n")
        f.write(f"  Images with detections: {stats['images_with_detections']}\n")
        f.write(f"  Average detections per image: {stats['avg_detections_per_image']:.2f}\n")
        f.write(f"\nInference Time:\n")
        f.write(f"  Average: {stats['avg_inference_time']*1000:.2f} ms\n")
        f.write(f"  Std Dev: {stats['std_inference_time']*1000:.2f} ms\n")
        f.write(f"  Min: {stats['min_inference_time']*1000:.2f} ms\n")
        f.write(f"  Max: {stats['max_inference_time']*1000:.2f} ms\n")
        f.write(f"  Total: {stats['total_time']:.2f} s\n")
        f.write(f"  FPS: {stats['fps']:.2f}\n")
    
    print(f"\nResults saved to {results_file}")
    
    if args.save_results:
        print(f"Visualization images saved to {output_dir}")
    
    print("\nNote: YOLOv8 is a general-purpose object detection model.")
    print("It may detect various objects (not just fires) in the images.")
    print("For fire-specific detection, a custom-trained YOLOv8 model would be needed.")


if __name__ == '__main__':
    main()

