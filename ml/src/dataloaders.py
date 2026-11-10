"""Transforms and DataLoaders for HAM10000, ready for EfficientNet-B0 transfer learning."""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from ml.src.dataset import HAM10000Dataset

# EfficientNet-B0's ImageNet-pretrained weights expect inputs normalized with
# these exact statistics; using anything else would mismatch what the
# pretrained filters learned to expect and waste the transfer learning.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = (224, 224)


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        # Augmentation (train only, applied on the fly -- every time an image is
        # drawn it gets a new random variant). It matters most for the rare
        # classes: the WeightedRandomSampler shows the same few df/vasc images
        # many times per epoch, and with weak augmentation v2 memorized them
        # (train bal_acc 0.97 vs val 0.65 late in fine-tuning). Since v3:
        # - RandomResizedCrop instead of a plain Resize: a random part of the
        #   image at a random scale. scale>=0.6 keeps the lesion (usually
        #   centred) in frame; ratio around 4:3 matches HAM10000's 600x450.
        # - Flips and a wider rotation (+-45 deg, was 20): dermatoscopic images
        #   have no canonical orientation, so these are label-preserving.
        # - Stronger ColorJitter (0.2, was 0.1) to mimic different dermatoscopes
        #   and lighting; hue only slightly (0.02), since colour itself is a
        #   diagnostic feature (e.g. for melanoma).
        # v1/v2 used: Resize, flips, RandomRotation(20), ColorJitter(0.1, 0.1, 0.1).
        ops = [
            transforms.RandomResizedCrop(
                INPUT_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    else:
        # Val/test: deterministic, unchanged since v1, so results stay comparable.
        ops = [transforms.Resize(INPUT_SIZE)]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def compute_sample_weights(dataset: HAM10000Dataset) -> torch.Tensor:
    """Per-sample weights for WeightedRandomSampler, inversely proportional to
    class frequency, so rare classes (e.g. df/vasc at ~1% of train) get
    oversampled within each epoch instead of being drowned out by nv (~67%).
    """
    class_counts = dataset.df["dx"].value_counts()
    sample_weights = dataset.df["dx"].map(lambda dx: 1.0 / class_counts[dx])
    return torch.tensor(sample_weights.values, dtype=torch.double)


def get_dataloaders(batch_size: int = 32, num_workers: int = 0):
    # num_workers=0 avoids Windows multiprocessing pitfalls; if you raise it,
    # the calling script needs an `if __name__ == "__main__":` guard.
    train_ds = HAM10000Dataset("train", transform=build_transforms(train=True))
    val_ds = HAM10000Dataset("val", transform=build_transforms(train=False))
    test_ds = HAM10000Dataset("test", transform=build_transforms(train=False))

    sampler = WeightedRandomSampler(
        compute_sample_weights(train_ds), num_samples=len(train_ds), replacement=True
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dl, val_dl, test_dl
