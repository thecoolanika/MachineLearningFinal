"""
ResNet-50 model for binary classification (fire vs. non-fire).

This module wraps torchvision's ResNet-50 implementation and adapts the
final classifier layer for our binary classification task.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    """
    ResNet-50 model adapted for binary classification.

    By default, the model is initialized with ImageNet-pretrained weights and
    the classifier head is replaced with a new linear layer for the desired
    number of classes.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: Optional[float] = None,
    ) -> None:
        """
        Initialize the ResNet-50 model.

        Args:
            num_classes: Number of output classes (2 for binary classification).
            pretrained: Whether to initialize with ImageNet-pretrained weights.
            dropout: Optional dropout probability to use in the classifier
                     head. If None, no dropout is used.
        """
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Replace classifier head
        in_features = self.backbone.fc.in_features

        if dropout is not None:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

