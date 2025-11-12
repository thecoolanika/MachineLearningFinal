"""
Preprocessing and augmentation utilities for image data.
"""
import torchvision.transforms as transforms
from typing import Optional


def get_transforms(
    split: str = 'train',
    image_size: int = 224,
    use_augmentation: bool = True
) -> transforms.Compose:
    """
    Get transform pipeline for images.
    
    Args:
        split: One of 'train', 'val', or 'test'
        image_size: Target image size (assumes square images)
        use_augmentation: Whether to use data augmentation (only for training)
        
    Returns:
        Compose transform pipeline
    """
    if split == 'train' and use_augmentation:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),  # Slightly larger for cropping
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform


def get_inverse_transform() -> transforms.Compose:
    """
    Get inverse transform to convert normalized tensors back to images for visualization.
    
    Returns:
        Compose transform pipeline
    """
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])

