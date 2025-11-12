"""
Exploratory Data Analysis for FlameVision dataset.

This script:
1. Checks dataset structure
2. Loads sample images from fire and non-fire classes
3. Displays statistics about the dataset
4. Visualizes sample images
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import FlameVisionDataset


def check_dataset_structure(data_dir: str):
    """
    Check if the dataset structure is correct.
    
    Args:
        data_dir: Root directory of the dataset
    """
    data_dir = Path(data_dir)
    
    print("=" * 60)
    print("Checking Dataset Structure")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    classes = ['fire', 'non_fire']
    
    structure_ok = True
    
    for split in splits:
        split_dir = data_dir / split
        print(f"\n{split.upper()} split:")
        
        if not split_dir.exists():
            print(f"  ❌ Directory {split_dir} does not exist")
            structure_ok = False
            continue
        
        for class_name in classes:
            class_dir = split_dir / class_name
            if class_dir.exists():
                # Count images
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                print(f"  ✓ {class_name}: {len(images)} images")
            else:
                print(f"  ❌ Directory {class_dir} does not exist")
                structure_ok = False
    
    if structure_ok:
        print("\n✓ Dataset structure is correct!")
    else:
        print("\n❌ Dataset structure has issues. Please check the directory structure.")
    
    return structure_ok


def get_dataset_statistics(data_dir: str):
    """
    Get statistics about the dataset.
    
    Args:
        data_dir: Root directory of the dataset
        
    Returns:
        Dictionary with statistics
    """
    data_dir = Path(data_dir)
    splits = ['train', 'val', 'test']
    classes = ['fire', 'non_fire']
    
    stats = {}
    
    for split in splits:
        stats[split] = {}
        split_dir = data_dir / split
        
        if not split_dir.exists():
            continue
        
        for class_name in classes:
            class_dir = split_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                stats[split][class_name] = len(images)
            else:
                stats[split][class_name] = 0
    
    return stats


def visualize_sample_images(data_dir: str, num_samples: int = 8):
    """
    Visualize sample images from fire and non-fire classes.
    
    Args:
        data_dir: Root directory of the dataset
        num_samples: Number of samples to display per class
    """
    data_dir = Path(data_dir)
    
    # Load sample images from training set
    train_dir = data_dir / 'train'
    
    if not train_dir.exists():
        print(f"Training directory {train_dir} does not exist")
        return
    
    # Get fire images
    fire_dir = train_dir / 'fire'
    non_fire_dir = train_dir / 'non_fire'
    
    fire_images = []
    non_fire_images = []
    
    if fire_dir.exists():
        fire_paths = list(fire_dir.glob('*.png')) + list(fire_dir.glob('*.jpg'))
        fire_images = fire_paths[:num_samples]
    
    if non_fire_dir.exists():
        non_fire_paths = list(non_fire_dir.glob('*.png')) + list(non_fire_dir.glob('*.jpg'))
        non_fire_images = non_fire_paths[:num_samples]
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
    fig.suptitle('Sample Images from FlameVision Dataset', fontsize=16)
    
    # Display fire images
    for idx, img_path in enumerate(fire_images):
        if idx < num_samples:
            try:
                img = Image.open(img_path)
                axes[0, idx].imshow(img)
                axes[0, idx].set_title('Fire', fontsize=10)
                axes[0, idx].axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    # Hide empty subplots for fire
    for idx in range(len(fire_images), num_samples):
        axes[0, idx].axis('off')
    
    # Display non-fire images
    for idx, img_path in enumerate(non_fire_images):
        if idx < num_samples:
            try:
                img = Image.open(img_path)
                axes[1, idx].imshow(img)
                axes[1, idx].set_title('Non-Fire', fontsize=10)
                axes[1, idx].axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    # Hide empty subplots for non-fire
    for idx in range(len(non_fire_images), num_samples):
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
    print(f"\nSample images saved to {output_dir / 'sample_images.png'}")
    plt.close()


def plot_dataset_statistics(data_dir: str):
    """
    Plot dataset statistics.
    
    Args:
        data_dir: Root directory of the dataset
    """
    stats = get_dataset_statistics(data_dir)
    
    # Prepare data for plotting
    splits = ['train', 'val', 'test']
    fire_counts = [stats.get(split, {}).get('fire', 0) for split in splits]
    non_fire_counts = [stats.get(split, {}).get('non_fire', 0) for split in splits]
    
    # Create bar plot
    x = np.arange(len(splits))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, fire_counts, width, label='Fire', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, non_fire_counts, width, label='Non-Fire', color='blue', alpha=0.7)
    
    ax.set_xlabel('Split')
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Distribution by Split and Class')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'dataset_statistics.png', dpi=150, bbox_inches='tight')
    print(f"Dataset statistics saved to {output_dir / 'dataset_statistics.png'}")
    plt.close()


def analyze_image_sizes(data_dir: str, num_samples: int = 100):
    """
    Analyze image sizes in the dataset.
    
    Args:
        data_dir: Root directory of the dataset
        num_samples: Number of samples to analyze
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    
    if not train_dir.exists():
        print(f"Training directory {train_dir} does not exist")
        return
    
    widths = []
    heights = []
    
    # Sample images from both classes
    for class_dir in [train_dir / 'fire', train_dir / 'non_fire']:
        if class_dir.exists():
            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            for img_path in images[:num_samples//2]:
                try:
                    img = Image.open(img_path)
                    widths.append(img.width)
                    heights.append(img.height)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    
    if len(widths) == 0:
        print("No images found to analyze")
        return
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Image Size Statistics")
    print("=" * 60)
    print(f"Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.2f}, Median: {np.median(widths):.2f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.2f}, Median: {np.median(heights):.2f}")
    
    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Image Width Distribution')
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(heights, bins=20, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Image Height Distribution')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'image_sizes.png', dpi=150, bbox_inches='tight')
    print(f"Image size analysis saved to {output_dir / 'image_sizes.png'}")
    plt.close()


def main():
    """Main function for exploratory data analysis."""
    # Set data directory
    data_dir = Path(__file__).parent.parent / 'data'
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist")
        print("Please download the FlameVision dataset and extract it to the 'data' directory")
        print("Dataset URL: https://data.mendeley.com/datasets/fgvscdjsmt/4")
        return
    
    print("=" * 60)
    print("Exploratory Data Analysis - FlameVision Dataset")
    print("=" * 60)
    
    # Check dataset structure
    if not check_dataset_structure(data_dir):
        return
    
    # Get and print statistics
    stats = get_dataset_statistics(data_dir)
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    for split in ['train', 'val', 'test']:
        if split in stats:
            fire_count = stats[split].get('fire', 0)
            non_fire_count = stats[split].get('non_fire', 0)
            total = fire_count + non_fire_count
            print(f"\n{split.upper()}:")
            print(f"  Fire: {fire_count}")
            print(f"  Non-Fire: {non_fire_count}")
            print(f"  Total: {total}")
    
    # Visualize sample images
    print("\n" + "=" * 60)
    print("Visualizing Sample Images")
    print("=" * 60)
    visualize_sample_images(data_dir, num_samples=8)
    
    # Plot dataset statistics
    print("\n" + "=" * 60)
    print("Plotting Dataset Statistics")
    print("=" * 60)
    plot_dataset_statistics(data_dir)
    
    # Analyze image sizes
    print("\n" + "=" * 60)
    print("Analyzing Image Sizes")
    print("=" * 60)
    analyze_image_sizes(data_dir, num_samples=100)
    
    print("\n" + "=" * 60)
    print("Exploratory Data Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {Path(__file__).parent.parent / 'results'}")


if __name__ == '__main__':
    main()

