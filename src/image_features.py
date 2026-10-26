"""Deterministic image transformations used by the drawing classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin


class GrayscaleVectorizer(BaseEstimator, TransformerMixin):
    """Convert image paths into fixed-size, normalized grayscale vectors.

    The transformer performs the same independent resize on each image, so it
    learns no dataset statistics and can safely be placed inside an sklearn
    pipeline. Images are stretched to a square, which is the preserved baseline
    behavior and may distort non-square drawings.
    """

    def __init__(self, size: int = 32) -> None:
        """Store the width and height, in pixels, for each output image."""
        self.size = size

    def fit(self, X: Any, y: Any = None) -> GrayscaleVectorizer:
        """Validate configuration; no data-dependent fitting is required."""
        if self.size < 8:
            raise ValueError("Image size must be at least 8 pixels")
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Load paths and return one flattened [0, 1] grayscale row per image."""
        paths = np.asarray(X, dtype=object)
        # ColumnTransformer may pass a one-column table; reduce it to paths.
        if paths.ndim == 2:
            paths = paths[:, 0]
        if paths.ndim != 1:
            raise ValueError("Expected one image path per row")

        features = []
        for path in paths:
            with Image.open(path) as image:
                # Correct camera orientation, discard color, and resize using
                # high-quality Lanczos filtering before scaling byte values.
                image = ImageOps.exif_transpose(image).convert("L")
                image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
                features.append(np.asarray(image, dtype=np.float32).ravel() / 255.0)
        return np.vstack(features)
