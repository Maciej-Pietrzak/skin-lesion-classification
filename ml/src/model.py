"""EfficientNet models (B0, B2) for HAM10000 classification via transfer learning."""

from typing import Callable, NamedTuple

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    WeightsEnum,
    efficientnet_b0,
    efficientnet_b2,
)

from ml.src.dataset import CLASSES

NUM_CLASSES = len(CLASSES)


class Architecture(NamedTuple):
    builder: Callable[..., nn.Module]
    weights: WeightsEnum
    # Side of the square input image in pixels.
    input_size: int


# Supported backbones. B2 is B0 scaled up in depth (16 -> 23 MBConv blocks),
# width (1280 -> 1408 final channels) and input resolution ("compound
# scaling"). 260x260 is B2's native resolution from the EfficientNet paper;
# torchvision evaluated its B2 weights at 288, but on this machine's 4 GB GPU
# 288 is ~2.7x slower per batch than 260, for a small expected gain.
# All use ImageNet IMAGENET1K_V1 weights, which is what dataloaders.py's
# IMAGENET_MEAN/STD normalization is matched to.
ARCHITECTURES = {
    "efficientnet_b0": Architecture(efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 224),
    "efficientnet_b2": Architecture(efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, 260),
}


def input_size(arch: str) -> tuple[int, int]:
    """(height, width) the given architecture is trained and run at."""
    side = ARCHITECTURES[arch].input_size
    return (side, side)


def build_model(arch: str, freeze_backbone: bool = False) -> nn.Module:
    spec = ARCHITECTURES[arch]
    model = spec.builder(weights=spec.weights)

    if freeze_backbone:
        # Freezes the pretrained feature extractor, leaving only the (new,
        # randomly-initialized) classifier head trainable. Useful as a first
        # training phase on a small dataset: large early gradients from the
        # untrained head would otherwise wreck the pretrained features before
        # they get a chance to adapt gradually.
        for param in model.features.parameters():
            param.requires_grad = False

    # classifier is Sequential(Dropout, Linear(features, 1000)) for every
    # EfficientNet variant (B0: 1280 features, B2: 1408); swap the 1000-way
    # ImageNet head for our 7 lesion classes.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model
