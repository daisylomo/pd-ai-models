"""Regression tests for image vectorization and drawing group splits."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.image_features import GrayscaleVectorizer
from src.train_drawing_models import make_grouped_splits


class ImageFeatureTests(unittest.TestCase):
    """Check output shape, numeric type, and range of grayscale vectors."""

    def test_vectorizer_returns_normalized_fixed_size_grayscale(self):
        """Ensure each image becomes a fixed-length vector scaled to [0, 1]."""
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (12, 10), color=(255, 0, 0)).save(image_path)
            vectorizer = GrayscaleVectorizer(size=8).fit([str(image_path)])
            values = vectorizer.transform([str(image_path)])
        self.assertEqual(values.shape, (1, 64))
        self.assertTrue(np.issubdtype(values.dtype, np.floating))
        self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))


class GroupedSplitTests(unittest.TestCase):
    """Check that drawing partitions retain both classes and isolate groups."""

    @staticmethod
    def _examples():
        """Create balanced toy rows with repeated observations per group."""
        labels, groups = [], []
        for label, prefix in ((0, "healthy"), (1, "parkinson")):
            for group_index in range(10):
                for _ in range(2):
                    labels.append(label)
                    groups.append(f"{prefix}:{group_index}")
        return np.asarray(labels), np.asarray(groups)

    def test_drawing_partitions_keep_groups_disjoint(self):
        """Ensure train, validation, and test contain disjoint candidate groups."""
        labels, groups = self._examples()
        partitions = make_grouped_splits(labels, groups, seed=42)
        group_sets = [set(groups[indices]) for indices in partitions]
        self.assertTrue(all(len(np.unique(labels[indices])) == 2 for indices in partitions))
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])


if __name__ == "__main__":
    # Allow this test module to be run directly as well as through unittest discovery.
    unittest.main()
