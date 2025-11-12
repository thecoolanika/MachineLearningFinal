"""
Test script to verify that the project setup is correct.

This script tests:
1. Imports
2. Model creation
3. Data loading utilities
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all imports work correctly."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from PIL import Image
        from sklearn.metrics import accuracy_score
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_model():
    """Test if the baseline model can be created."""
    print("\nTesting model creation...")
    try:
        from src.models import BaselineCNN
        from src.models.baseline_cnn import count_parameters
        
        model = BaselineCNN(num_classes=2, input_channels=3)
        num_params = count_parameters(model)
        print(f"✓ Model created successfully")
        print(f"  Number of parameters: {num_params:,}")
        
        # Test forward pass
        import torch
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test if data loading utilities work."""
    print("\nTesting data loading utilities...")
    try:
        from src.data_loader import FlameVisionDataset
        from src.preprocessing import get_transforms
        
        # Test transforms
        transform = get_transforms(split='train', image_size=224, use_augmentation=True)
        print("✓ Transforms created successfully")
        
        # Test dataset (will fail if dataset doesn't exist, which is OK)
        data_dir = Path(__file__).parent.parent / 'data'
        if data_dir.exists():
            try:
                dataset = FlameVisionDataset(data_dir=str(data_dir), split='train', transform=transform)
                print(f"✓ Dataset loaded successfully: {len(dataset)} images")
            except Exception as e:
                print(f"⚠️  Dataset not found or incomplete: {e}")
                print("  This is OK if you haven't downloaded the dataset yet")
        else:
            print("⚠️  Data directory does not exist")
            print("  This is OK if you haven't downloaded the dataset yet")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test if utility functions work."""
    print("\nTesting utility functions...")
    try:
        from src.utils import calculate_metrics
        import numpy as np
        
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = calculate_metrics(y_true, y_pred)
        print("✓ Utility functions work correctly")
        print(f"  Sample metrics: {metrics}")
        return True
    except Exception as e:
        print(f"❌ Utility functions error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Project Setup")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Model", test_model()))
    results.append(("Data Loader", test_data_loader()))
    results.append(("Utils", test_utils()))
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All tests passed! Project setup is correct.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

