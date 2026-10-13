import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.image_features import GrayscaleVectorizer
from src.train_drawing_models import make_grouped_splits
from src.train_handwriting_models import make_splits, participant_group, summarize_task


class ImageFeatureTests(unittest.TestCase):
    def test_vectorizer_returns_normalized_fixed_size_grayscale(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (12, 10), color=(255, 0, 0)).save(image_path)

            vectorizer = GrayscaleVectorizer(size=8).fit([str(image_path)])
            values = vectorizer.transform([str(image_path)])

        self.assertEqual(values.shape, (1, 64))
        self.assertTrue(np.issubdtype(values.dtype, np.floating))
        self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))


class HandwritingFeatureTests(unittest.TestCase):
    def test_task_summary_calculates_path_geometry(self):
        points = pd.DataFrame({
            "X": [0.0, 3.0, 3.0],
            "Y": [0.0, 4.0, 8.0],
            "Z": [1.0, 2.0, 3.0],
            "Pressure": [0.0, 0.5, 1.0],
            "GripAngle": [10.0, 20.0, 30.0],
        })

        summary = summarize_task(points)

        self.assertEqual(summary["n_points"], 3.0)
        self.assertAlmostEqual(summary["path_length"], 9.0)
        self.assertAlmostEqual(summary["net_displacement"], np.sqrt(73.0))
        self.assertAlmostEqual(summary["path_efficiency"], np.sqrt(73.0) / 9.0)

    def test_filename_group_conventions_are_explicit(self):
        self.assertEqual(participant_group(Path("C_0007.txt"), "control"), "control:C_0007")
        self.assertEqual(participant_group(Path("P_02100003.txt"), "parkinson"), "parkinson:P_0210")


class GroupedSplitTests(unittest.TestCase):
    @staticmethod
    def _examples():
        labels, groups = [], []
        for label, prefix in ((0, "control"), (1, "parkinson")):
            for group_index in range(10):
                for _ in range(2):
                    labels.append(label)
                    groups.append(f"{prefix}:{group_index}")
        return np.asarray(labels), np.asarray(groups)

    def _assert_group_disjoint(self, split_function):
        labels, groups = self._examples()
        partitions = split_function(labels, groups, seed=42)
        group_sets = [set(groups[indices]) for indices in partitions]
        self.assertTrue(all(len(np.unique(labels[indices])) == 2 for indices in partitions))
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])

    def test_drawing_partitions_keep_groups_separate(self):
        self._assert_group_disjoint(make_grouped_splits)

    def test_handwriting_partitions_keep_groups_separate(self):
        self._assert_group_disjoint(make_splits)


if __name__ == "__main__":
    unittest.main()
