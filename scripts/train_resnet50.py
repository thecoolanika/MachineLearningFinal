"""
Training script for ResNet-50 model.

This script trains a ResNet-50 model for binary classification
(fire vs. non-fire) on the FlameVision dataset.
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_data_loaders
from src.models.resnet50 import ResNet50, count_parameters
from src.utils import evaluate_model, save_checkpoint


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training set
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            {"loss": running_loss / len(train_loader), "acc": 100 * correct / total}
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ResNet-50 model")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Root directory of the dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (typically lower for pretrained models)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size for training (ResNet-50 expects 224x224)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=True,
        help="Use data augmentation for training",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet pretraining for ResNet-50",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True, parents=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir / "resnet50"))

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        use_augmentation=args.use_augmentation,
    )

    # Create model
    print("Creating model...")
    model = ResNet50(
        num_classes=2,
        pretrained=not args.no_pretrained,
    )

    # Freeze backbone; train only final classifier to reduce overfitting on small data
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # Count trainable parameters (should be small after freezing)
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")

    # Loss function and optimizer (optimize only trainable parameters)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_f1 = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1 = checkpoint["metrics"].get("f1_score", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        print("Validating...")
        val_metrics, _, _ = evaluate_model(model, val_loader, device, criterion)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"] * 100
        val_f1 = val_metrics["f1_score"]

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val F1-Score: {val_f1:.4f}")

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("F1-Score/Val", val_f1, epoch)
        writer.add_scalar("Precision/Val", val_metrics["precision"], epoch)
        writer.add_scalar("Recall/Val", val_metrics["recall"], epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_path = save_dir / "resnet50_best.pth"
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics, str(best_model_path)
            )
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f"resnet50_epoch_{epoch + 1}.pth"
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics, str(checkpoint_path)
            )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    # Load best model
    best_model_path = save_dir / "resnet50_best.pth"
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_metrics, test_labels, test_preds = evaluate_model(
        model, test_loader, device, criterion
    )

    print("\nTest Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")

    # Save test results
    results_file = results_dir / "resnet50_test_results.txt"
    with open(results_file, "w") as f:
        f.write("ResNet-50 - Test Set Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Accuracy: {test_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {test_metrics['f1_score']:.4f}\n")

    print(f"\nTest results saved to {results_file}")

    # Plot training curves
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curve
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curve
    axes[1].plot(train_accs, label="Train Acc")
    axes[1].plot(val_accs, label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        results_dir / "resnet50_training_curves.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(
        f"Training curves saved to {results_dir / 'resnet50_training_curves.png'}"
    )
    plt.close()

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

