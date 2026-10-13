"""Deterministic image transformations used by the drawing classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, TransformerMixin


class GrayscaleVectorizer(BaseEstimator, TransformerMixin):
    """Convert image paths into fixed-size, normalized grayscale vectors."""

    def __init__(self, size: int = 32) -> None:
        self.size = size

    def fit(self, X: Any, y: Any = None) -> GrayscaleVectorizer:
        if self.size < 8:
            raise ValueError("Image size must be at least 8 pixels")
        return self

    def transform(self, X: Any) -> np.ndarray:
        paths = np.asarray(X, dtype=object)
        if paths.ndim == 2:
            paths = paths[:, 0]
        if paths.ndim != 1:
            raise ValueError("Expected one image path per row")

        features = []
        for path in paths:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("L")
                image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
                features.append(np.asarray(image, dtype=np.float32).ravel() / 255.0)
        return np.vstack(features)
