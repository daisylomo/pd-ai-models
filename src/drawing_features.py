"""Deterministic, image-only representations for spiral and wave drawings.

All image operations are applied independently to each image. The sklearn
transformers do not learn dataset statistics; downstream scaling remains in
the estimator pipeline and is therefore fitted within each training fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin


def _load_gray(path: str, size: int, preserve_aspect: bool) -> np.ndarray:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        if preserve_aspect:
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("L", (size, size), color=255)
            canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
            image = canvas
        else:
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def _otsu_threshold(image: np.ndarray) -> float:
    values = np.clip(np.rint(image * 255), 0, 255).astype(np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(float)
    probability = histogram / histogram.sum()
    omega = np.cumsum(probability)
    means = np.cumsum(probability * np.arange(256))
    total = means[-1]
    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 0
    between[valid] = (total * omega[valid] - means[valid]) ** 2 / denominator[valid]
    return float(np.argmax(between) / 255.0)


def _handcrafted(image: np.ndarray) -> np.ndarray:
    """Return occupancy, shape moments, radial occupancy and gradient bins."""
    size = image.shape[0]
    threshold = _otsu_threshold(image)
    foreground = image < threshold
    # Faint scans can place the Otsu threshold at an extreme. An image-specific
    # lower-tail fallback keeps features finite and avoids a learned threshold.
    if foreground.sum() < max(4, int(image.size * 0.002)):
        foreground = image < float(np.quantile(image, 0.15))

    height, width = foreground.shape
    cell_features: list[float] = []
    grid = 4
    for row in range(grid):
        for column in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = column * width // grid, (column + 1) * width // grid
            cell_features.append(float(foreground[y0:y1, x0:x1].mean()))

    ys, xs = np.nonzero(foreground)
    if len(xs):
        x = xs.astype(float) / max(1, width - 1)
        y = ys.astype(float) / max(1, height - 1)
        cx, cy = float(x.mean()), float(y.mean())
        dx, dy = x - cx, y - cy
        moments = [cx, cy, float(np.mean(dx * dx)), float(np.mean(dy * dy)), float(np.mean(dx * dy))]
        radius = np.sqrt(dx * dx + dy * dy) / np.sqrt(2.0)
        radial, _ = np.histogram(radius, bins=12, range=(0.0, 1.0))
        radial = radial.astype(float) / len(xs)
    else:
        moments = [0.0] * 5
        radial = np.zeros(12, dtype=float)

    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    orientation: list[float] = []
    for row in range(grid):
        for column in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = column * width // grid, (column + 1) * width // grid
            hist, _ = np.histogram(
                angle[y0:y1, x0:x1], bins=8, range=(0.0, np.pi),
                weights=magnitude[y0:y1, x0:x1],
            )
            total = float(hist.sum())
            orientation.extend((hist / total if total > 0 else hist).tolist())

    return np.asarray(
        [float(foreground.mean()), float(image.std()), *cell_features, *moments, *radial, *orientation],
        dtype=np.float32,
    )


class DrawingImageVectorizer(BaseEstimator, TransformerMixin):
    """Vectorize aspect-preserving grayscale pixels and/or descriptors."""

    def __init__(self, representation: str = "aspect_pixels", size: int = 64) -> None:
        self.representation = representation
        self.size = size

    def fit(self, X: Any, y: Any = None) -> DrawingImageVectorizer:
        if self.representation not in {"aspect_pixels", "handcrafted", "combined"}:
            raise ValueError(f"Unknown drawing representation: {self.representation}")
        if self.size < 16:
            raise ValueError("Image size must be at least 16 pixels")
        return self

    def transform(self, X: Any) -> np.ndarray:
        paths = np.asarray(X, dtype=object)
        if paths.ndim == 2:
            paths = paths[:, 0]
        if paths.ndim != 1:
            raise ValueError("Expected one image path per row")
        rows = []
        for path in paths:
            image = _load_gray(str(path), self.size, preserve_aspect=True)
            descriptor = _handcrafted(image)
            if self.representation == "handcrafted":
                row = descriptor
            elif self.representation == "combined":
                row = np.concatenate([image.ravel(), descriptor])
            else:
                row = image.ravel()
            rows.append(row)
        if not rows:
            width = self.size * self.size + (163 if self.representation == "combined" else 0)
            return np.empty((0, width), dtype=np.float32)
        return np.vstack(rows).astype(np.float32, copy=False)
