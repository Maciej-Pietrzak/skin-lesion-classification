"""Transforms and DataLoaders for HAM10000, ready for EfficientNet transfer learning."""

import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from ml.src.dataset import DATA_DIR, HAM10000Dataset
from ml.src.lca import LowCostAugment

# "base" = the augmentation of v3 (resize) / v7 (native_crop).
# "lca" (since v8) = base + one Low-Cost-Augment sub-policy with p=0.3 on top
# (ml/src/lca.py); native_crop only.
AUGMENT_MODES = ("base", "lca")

# EfficientNet's ImageNet-pretrained weights (B0 and B2 alike) expect inputs
# normalized with these exact statistics; using anything else would mismatch
# what the pretrained filters learned to expect and waste the transfer learning.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


PREPROCESS_MODES = ("resize", "native_crop")

# Number of crops per side of the evaluation grid in "native_crop" mode:
# 4x4 = 16 crops, as in Shen et al. 2021 (arXiv:2101.02353) / Gessert et al. 2018.
GRID_CROPS_PER_SIDE = 4


class RandomScale:
    """Rescales the whole image by a random factor, keeping its aspect ratio.

    Used in "native_crop" mode instead of RandomResizedCrop: the image stays
    at (roughly) its original 600x450 resolution, and the scale jitter makes
    the model robust to lesions of different sizes. With factor 0.6 the
    image is 360x270, still larger than a 224x224 crop.
    """

    def __init__(self, scale: tuple[float, float]):
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(*self.scale)
        w, h = img.size
        return img.resize((round(w * factor), round(h * factor)), Image.BILINEAR)


class GridCrops:
    """Cuts n x n equally spaced crops of `size` from the unscaled image,
    from the top-left to the bottom-right corner, and returns them as one
    (n*n, C, H, W) tensor. The model's predictions for all crops are then
    averaged (multi-crop evaluation / test-time augmentation).
    """

    def __init__(self, size: tuple[int, int], per_side: int, to_tensor: transforms.Compose):
        self.size = size
        self.per_side = per_side
        self.to_tensor = to_tensor

    @staticmethod
    def _offsets(full: int, crop: int, n: int) -> list[int]:
        if full <= crop or n == 1:
            return [max(0, (full - crop) // 2)] * n
        return [round(i * (full - crop) / (n - 1)) for i in range(n)]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        crop_h, crop_w = self.size
        w, h = img.size
        if w < crop_w or h < crop_h:
            # Smaller than one crop (doesn't happen in HAM10000): upscale first.
            img = transforms.Resize(max(crop_h, crop_w))(img)
            w, h = img.size
        crops = [
            img.crop((x, y, x + crop_w, y + crop_h))
            for y in self._offsets(h, crop_h, self.per_side)
            for x in self._offsets(w, crop_w, self.per_side)
        ]
        return torch.stack([self.to_tensor(c) for c in crops])


def build_transforms(
    train: bool,
    input_size: tuple[int, int],
    preprocess: str = "resize",
    multi_crop: bool = True,
    augment: str = "base",
    pairing_paths: list | None = None,
) -> transforms.Compose:
    """input_size is the model's (height, width) -- model.input_size(arch):
    224x224 for B0, 260x260 for B2. It has no default on purpose: feeding a
    model images at a size it wasn't trained at silently degrades predictions.

    preprocess:
    - "resize" (v1-v6): the whole image is shrunk to input_size. One image ->
      one (C, H, W) tensor.
    - "native_crop" (since v7): no shrinking. Train: random rescale 0.6-1.4,
      then a random input_size crop. Val/test with multi_crop=True: GridCrops,
      one image -> a (16, C, H, W) tensor whose predictions must be averaged.
      With multi_crop=False: a single centre crop (C, H, W) -- 16x cheaper,
      used for validation after every training epoch, where it only has to
      rank checkpoints, not give the final score.
    multi_crop is ignored in "resize" mode.

    augment (train only): "base" or "lca". "lca" adds a LowCostAugment step
    after the base augmentation and before the crop, so it works on the
    full-resolution image as in the paper; pairing_paths = training image
    files for its sample-pairing operation.
    """
    if preprocess not in PREPROCESS_MODES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_MODES}, got {preprocess!r}")
    if augment not in AUGMENT_MODES:
        raise ValueError(f"augment must be one of {AUGMENT_MODES}, got {augment!r}")
    if train and augment == "lca" and preprocess != "native_crop":
        raise ValueError('augment="lca" is only implemented for preprocess="native_crop"')

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if preprocess == "native_crop":
        if train:
            # Same flips / rotation / colour as v3, only the geometry differs:
            # rotate the full image first, then crop, so the crop rarely
            # reaches the black corners the rotation leaves behind.
            ops = [
                RandomScale((0.6, 1.4)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ]
            if augment == "lca":
                ops.append(LowCostAugment(pairing_paths or []))
            # pad_if_needed: LCA's own Scale (0.6x) on top of RandomScale
            # (0.6x) can leave a 216x162 image, smaller than the crop; it is
            # then padded with black. Never triggers without LCA (min 360x270),
            # so v7 is unaffected.
            ops += [transforms.RandomCrop(input_size, pad_if_needed=True), to_tensor]
            return transforms.Compose(ops)
        if multi_crop:
            return GridCrops(input_size, GRID_CROPS_PER_SIDE, to_tensor)
        return transforms.Compose([transforms.CenterCrop(input_size), to_tensor])

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
                input_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    else:
        # Val/test: deterministic, unchanged since v1, so results stay comparable.
        ops = [transforms.Resize(input_size)]

    ops.append(to_tensor)
    return transforms.Compose(ops)


def compute_sample_weights(dataset: HAM10000Dataset) -> torch.Tensor:
    """Per-sample weights for WeightedRandomSampler, inversely proportional to
    class frequency, so rare classes (e.g. df/vasc at ~1% of train) get
    oversampled within each epoch instead of being drowned out by nv (~67%).
    """
    class_counts = dataset.df["dx"].value_counts()
    sample_weights = dataset.df["dx"].map(lambda dx: 1.0 / class_counts[dx])
    return torch.tensor(sample_weights.values, dtype=torch.double)


def get_dataloaders(
    input_size: tuple[int, int],
    batch_size: int = 32,
    num_workers: int = 2,
    preprocess: str = "resize",
    augment: str = "base",
):
    # With preprocess="native_crop", val is a single centre crop per image
    # (fast, checked every epoch) and test is the 16-crop grid, so test
    # batches have shape (B, 16, C, H, W) -- see build_transforms() and
    # evaluate.predict_proba().
    # num_workers = background processes that load + augment the next batches
    # while the GPU works on the current one. With 0 (v1-v4) everything ran in
    # one process and the GPU sat idle during JPEG decoding and augmentation,
    # which got noticeably slower with v3's heavier augmentation. Each worker is
    # a separate Python process (~0.3-0.5 GB RAM), so this stays modest -- this
    # machine has run out of RAM during training before. On Windows, every
    # script that calls this needs an `if __name__ == "__main__":` guard
    # (train/evaluate/gradcam have one). Results stay reproducible with a fixed
    # seed, but not bit-identical to runs with a different num_workers.
    loader_kwargs = {
        "num_workers": num_workers,
        # Keep workers alive between epochs instead of re-spawning them (slow
        # on Windows, where each worker re-imports torch).
        "persistent_workers": num_workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_ds = HAM10000Dataset("train")
    train_ds.transform = build_transforms(
        True,
        input_size,
        preprocess,
        augment=augment,
        pairing_paths=[DATA_DIR / p for p in train_ds.df["image_path"]],
    )
    val_ds = HAM10000Dataset(
        "val", transform=build_transforms(False, input_size, preprocess, multi_crop=False)
    )
    test_ds = HAM10000Dataset("test", transform=build_transforms(False, input_size, preprocess))

    sampler = WeightedRandomSampler(
        compute_sample_weights(train_ds), num_samples=len(train_ds), replacement=True
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_dl, val_dl, test_dl
