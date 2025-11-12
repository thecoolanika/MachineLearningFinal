"""
Evaluation script for baseline CNN model.

This script evaluates a trained model on the test set and generates
detailed metrics and visualizations.
"""
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_data_loaders
from src.models import BaselineCNN
from src.utils import evaluate_model, plot_confusion_matrix


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate baseline CNN model')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory of the dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading data...")
    _, _, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_augmentation=False
    )
    
    # Create model
    print("Creating model...")
    model = BaselineCNN(num_classes=2, input_channels=3)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_metrics, test_labels, test_preds = evaluate_model(
        model, test_loader, device, criterion
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Set Results")
    print("=" * 60)
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Save results
    results_file = output_dir / 'evaluation_results.txt'
    with open(results_file, 'w') as f:
        f.write("Baseline CNN - Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {test_metrics['f1_score']:.4f}\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        test_labels,
        test_preds,
        class_names=['Non-Fire', 'Fire'],
        save_path=str(output_dir / 'confusion_matrix.png')
    )
    print(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

