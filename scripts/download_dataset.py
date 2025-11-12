"""
Script to help download and verify the FlameVision dataset.

The dataset is available at:
https://data.mendeley.com/datasets/fgvscdjsmt/4

This script provides instructions and checks if the dataset is properly set up.
"""
from pathlib import Path


def check_dataset(data_dir: str = 'data') -> bool:
    """
    Check if the dataset is properly downloaded and extracted.
    
    Args:
        data_dir: Root directory of the dataset
        
    Returns:
        True if dataset is properly set up, False otherwise
    """
    data_dir = Path(data_dir)
    
    print("=" * 60)
    print("Checking FlameVision Dataset")
    print("=" * 60)
    
    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} does not exist")
        return False
    
    # Check for required splits
    splits = ['train', 'val', 'test']
    classes = ['fire', 'non_fire']
    
    all_ok = True
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"❌ Split directory {split_dir} does not exist")
            all_ok = False
            continue
        
        for class_name in classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"❌ Class directory {class_dir} does not exist")
                all_ok = False
                continue
            
            # Count images
            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            if len(images) == 0:
                print(f"⚠️  No images found in {class_dir}")
            else:
                print(f"✓ {split}/{class_name}: {len(images)} images")
    
    if all_ok:
        print("\n✓ Dataset is properly set up!")
        return True
    else:
        print("\n❌ Dataset is not properly set up")
        return False


def print_instructions():
    """Print instructions for downloading the dataset."""
    print("=" * 60)
    print("FlameVision Dataset Download Instructions")
    print("=" * 60)
    print("\n1. Visit the dataset page:")
    print("   https://data.mendeley.com/datasets/fgvscdjsmt/4")
    print("\n2. Download the dataset (you may need to create a Mendeley account)")
    print("\n3. Extract the dataset to the 'data' directory")
    print("\n4. The expected directory structure is:")
    print("   data/")
    print("   ├── train/")
    print("   │   ├── fire/")
    print("   │   └── non_fire/")
    print("   ├── val/")
    print("   │   ├── fire/")
    print("   │   └── non_fire/")
    print("   └── test/")
    print("       ├── fire/")
    print("       └── non_fire/")
    print("\n5. Run this script again to verify the setup")
    print("=" * 60)


def main():
    """Main function."""
    data_dir = Path('data')
    
    # Check if dataset exists
    if check_dataset(data_dir):
        print("\nDataset is ready to use!")
        print("You can now run:")
        print("  python scripts/exploratory_analysis.py")
        print("  python scripts/train_baseline.py")
    else:
        print_instructions()


if __name__ == '__main__':
    main()

