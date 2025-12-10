"""
Model architectures for wildfire detection.
"""
from .baseline_cnn import BaselineCNN
from .efficientnet_b0 import EfficientNetB0
from .resnet50 import ResNet50

__all__ = ['BaselineCNN', 'EfficientNetB0', 'ResNet50']


