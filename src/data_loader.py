"""
Data loading utilities for FlameVision dataset.
"""
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class FlameVisionDataset(Dataset):
    """
    Dataset class for FlameVision wildfire detection dataset.
    
    Args:
        data_dir: Root directory containing train/val/test splits
        split: One of 'train', 'val', or 'test'
        transform: Optional transform to be applied on images
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Set up paths
        self.split_dir = self.data_dir / split
        self.fire_dir = self.split_dir / 'fire'
        self.non_fire_dir = self.split_dir / 'non_fire'
        
        # Check if directories exist
        if not self.split_dir.exists():
            raise ValueError(f"Split directory {self.split_dir} does not exist")
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Load fire images (label = 1)
        if self.fire_dir.exists():
            fire_images = list(self.fire_dir.glob('*.png')) + list(self.fire_dir.glob('*.jpg'))
            self.image_paths.extend(fire_images)
            self.labels.extend([1] * len(fire_images))
        
        # Load non-fire images (label = 0)
        if self.non_fire_dir.exists():
            non_fire_images = list(self.non_fire_dir.glob('*.png')) + list(self.non_fire_dir.glob('*.jpg'))
            self.image_paths.extend(non_fire_images)
            self.labels.extend([0] * len(non_fire_images))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.split_dir}")
        
        print(f"Loaded {len(self.image_paths)} images from {split} split")
        print(f"  - Fire images: {sum(self.labels)}")
        print(f"  - Non-fire images: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get an image and its label.
        
        Args:
            idx: Index of the image
            
        Returns:
            Tuple of (image tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing train/val/test splits
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        image_size: Target image size (assumes square images)
        use_augmentation: Whether to use data augmentation for training set
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from src.preprocessing import get_transforms
    
    # Get transforms
    train_transform = get_transforms(
        split='train',
        image_size=image_size,
        use_augmentation=use_augmentation
    )
    val_transform = get_transforms(
        split='val',
        image_size=image_size,
        use_augmentation=False
    )
    test_transform = get_transforms(
        split='test',
        image_size=image_size,
        use_augmentation=False
    )
    
    # Create datasets
    train_dataset = FlameVisionDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )
    val_dataset = FlameVisionDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )
    test_dataset = FlameVisionDataset(
        data_dir=data_dir,
        split='test',
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

