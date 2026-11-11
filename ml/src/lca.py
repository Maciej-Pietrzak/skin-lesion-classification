"""Low-Cost-Augment (LCA) from Shen et al. 2021 / BME Frontiers 2022,
"A Low-Cost High-Performance Data Augmentation for Deep Learning-Based Skin
Lesion Classification" (arXiv:2101.02353, Tables 1 and X1).

The authors' code was never released (their GitHub repository only has a
README), so this is a re-implementation from the paper text. LCA is a fixed
menu of 12 sub-policies, each one colour operation paired with one geometric
operation. For every training image one sub-policy is drawn uniformly at
random and applied with probability p (best in the paper: 0.3); each
operation's magnitude is drawn uniformly from its range on every call.

Where the paper is ambiguous, the choices made here are:
- p applies to the whole sub-policy (both operations together), as the text
  describes it, not to each operation separately as in AutoAugment.
- Sample pairing blends in a random *training* image (the paper: another
  image from the same batch, which a per-image transform cannot see). The
  label is unchanged, as in the paper.
- Gaussian noise "with rate magnitude" [0, 0.4]: noise std = magnitude * 0.25
  on the 0-1 pixel scale (max std ~25/255). Taking 0.4 literally as the std
  would bury the lesion in noise.
- Shear "with rate magnitude" [-0.3, 0.3]: the affine shear coefficient,
  around the image centre (as in AutoAugment's ShearX/ShearY).
- Equalize_YUV: the paper says "each YUV channel", but equalizing the
  chroma channels (U/V) turns the image into red/cyan blotches, wiping out
  lesion colour, which is diagnostic. Converting to YUV only makes sense to
  equalize brightness without touching colour (the usual OpenCV idiom), so
  only Y is equalized. Plain Equalize (all RGB channels) is kept as written.
- Rotation and shear leave black corners, like torchvision's RandomRotation.

Applied on the full-resolution PIL image, before the random crop (the paper:
sub-policy on the 600x450 image, then a random 224x224 crop).
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# --- colour operations (magnitude ranges from Table X1) ---------------------


def _sample_pairing(img, pairing_paths):
    weight = random.uniform(0.0, 0.4)
    other = Image.open(random.choice(pairing_paths)).convert("RGB").resize(img.size)
    return Image.blend(img, other, weight)


def _gaussian_noise(img, _):
    std = random.uniform(0.0, 0.4) * 0.25 * 255
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.normal(0.0, std, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _solarize_add(img, _):
    # Pixels darker than 128 get brighter by `add`; the rest stay as they are.
    add = random.randint(1, 110)
    arr = np.asarray(img, dtype=np.int16)
    arr = np.where(arr < 128, np.minimum(arr + add, 255), arr)
    return Image.fromarray(arr.astype(np.uint8))


def _enhance(enhancer):
    # 1.0 = original image; 0.1-1.9 = from almost removed to almost doubled.
    return lambda img, _: enhancer(img).enhance(random.uniform(0.1, 1.9))


def _color_shift(img, _):
    shift = np.array([random.randint(-20, 20) for _ in range(3)], dtype=np.int16)
    arr = np.asarray(img, dtype=np.int16) + shift
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _equalize_yuv(img, _):
    # PIL's YCbCr is the digital form of YUV. Only the luminance (Y) is
    # equalized -- see the module docstring.
    y, cb, cr = img.convert("YCbCr").split()
    return Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB")


def _equalize(img, _):
    return ImageOps.equalize(img)  # each RGB channel separately


def _posterize(img, _):
    return ImageOps.posterize(img, random.randint(4, 8))


def _autocontrast(img, _):
    return ImageOps.autocontrast(img)


# --- geometric operations ---------------------------------------------------


def _rotate(img, _):
    return img.rotate(random.uniform(-30, 30), resample=Image.BILINEAR)


def _flip(img, _):
    return ImageOps.mirror(img) if random.random() < 0.5 else ImageOps.flip(img)


def _scale(img, _):
    factor = random.uniform(0.6, 1.4)
    w, h = img.size
    return img.resize((round(w * factor), round(h * factor)), Image.BILINEAR)


def _cutout(img, _):
    side = random.randint(0, 60)
    if side == 0:
        return img
    w, h = img.size
    x, y = random.randint(0, w - side), random.randint(0, h - side)
    img = img.copy()
    img.paste((128, 128, 128), (x, y, x + side, y + side))
    return img


def _shear(axis):
    def op(img, _):
        m = random.uniform(-0.3, 0.3)
        w, h = img.size
        # Affine maps output -> input pixel; the offset keeps the centre fixed.
        coeffs = (1, m, -m * h / 2, 0, 1, 0) if axis == "x" else (1, 0, 0, m, 1, -m * w / 2)
        return img.transform(img.size, Image.AFFINE, coeffs, resample=Image.BILINEAR)

    return op


# Table 1 of the paper: (colour operation, geometric operation).
SUB_POLICIES = [
    (_sample_pairing, _rotate),
    (_gaussian_noise, _flip),
    (_solarize_add, _cutout),
    (_enhance(ImageEnhance.Color), _shear("x")),
    (_enhance(ImageEnhance.Contrast), _shear("y")),
    (_enhance(ImageEnhance.Brightness), _scale),
    (_enhance(ImageEnhance.Sharpness), _rotate),
    (_color_shift, _scale),
    (_equalize_yuv, _shear("x")),
    (_posterize, _shear("y")),
    (_autocontrast, _flip),
    (_equalize, _cutout),
]


class LowCostAugment:
    """PIL -> PIL transform: one random LCA sub-policy, applied with probability p."""

    def __init__(self, pairing_paths: list[Path], p: float = 0.3):
        if not pairing_paths:
            raise ValueError("Sample pairing needs at least one training image path")
        self.pairing_paths = list(pairing_paths)
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        colour_op, geometric_op = random.choice(SUB_POLICIES)
        return geometric_op(colour_op(img, self.pairing_paths), self.pairing_paths)
