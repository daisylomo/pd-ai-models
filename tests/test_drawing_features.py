import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.drawing_features import DrawingImageVectorizer
from src.train_drawing_comparisons import _evaluate, _group_cv, predict_from_bundle
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class DrawingRepresentationTests(unittest.TestCase):
    def test_representations_are_finite_and_have_stable_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wave.png"
            image = Image.new("L", (90, 32), 245)
            draw = ImageDraw.Draw(image)
            draw.line([(0, 10), (20, 27), (40, 7), (60, 25), (89, 8)], fill=40, width=2)
            image.save(path)
            for representation, expected_width in (("aspect_pixels", 64 * 64), ("handcrafted", 163), ("combined", 64 * 64 + 163)):
                vector = DrawingImageVectorizer(representation=representation, size=64).fit_transform([str(path)])
                self.assertEqual(vector.shape, (1, expected_width))
                self.assertTrue(np.isfinite(vector).all())

    def test_handcrafted_transform_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spiral.png"
            image = Image.new("L", (48, 48), 255)
            draw = ImageDraw.Draw(image)
            draw.ellipse((8, 8, 40, 40), outline=30, width=2)
            image.save(path)
            transformer = DrawingImageVectorizer(representation="handcrafted", size=64).fit([str(path)])
            first = transformer.transform([str(path)])
            second = transformer.transform([str(path)])
        np.testing.assert_array_equal(first, second)

    def test_group_cv_keeps_groups_disjoint(self):
        rows = []
        for target in (0, 1):
            for group_index in range(6):
                for image_index in range(2):
                    rows.append((target, f"class{target}:g{group_index}", f"image-{target}-{group_index}-{image_index}"))
        frame = pd.DataFrame(rows, columns=["target", "group", "path"])
        y = frame["target"].to_numpy()
        groups = frame["group"].to_numpy()
        folds = _group_cv(frame[["path"]], y, groups, seed=42)
        for train, validation in folds:
            self.assertFalse(set(groups[train]) & set(groups[validation]))
            self.assertEqual(set(y[validation]), {0, 1})

    def test_metrics_report_fixed_class_support_and_confusion(self):
        model = Pipeline([("classifier", DummyClassifier(strategy="constant", constant=1))]).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        result = _evaluate(model, pd.DataFrame({"unused": [0, 1, 2, 3]}), np.asarray([0, 0, 1, 1]))
        self.assertEqual(result["class_support"], {"healthy": 2, "parkinson": 2})
        self.assertEqual(result["confusion_matrix_healthy_parkinson"], [[0, 2], [0, 2]])
        self.assertAlmostEqual(result["recall_parkinson"], 1.0)

    def test_saved_bundle_prediction_uses_image_extractor_and_task(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, color in enumerate((20, 40, 210, 230)):
                path = Path(directory) / f"sample{index}.png"
                Image.new("L", (20, 40), color=color).save(path)
                paths.append(path)
            extractor = DrawingImageVectorizer(representation="aspect_pixels", size=16).fit(paths)
            features = extractor.transform(paths)
            estimator = Pipeline([("classifier", LogisticRegression())]).fit(
                pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])]),
                np.asarray([0, 0, 1, 1]),
            )
            bundle = {
                "feature_extractor": extractor,
                "feature_columns": [f"feature_{i}" for i in range(features.shape[1])],
                "include_task": False,
                "pipeline": estimator,
            }
            predictions = predict_from_bundle(bundle, paths)
        self.assertEqual(predictions.shape, (4,))
        self.assertEqual(predictions.tolist(), [0, 0, 1, 1])

    def test_combined_saved_model_requires_task_for_each_image(self):
        with self.assertRaises(ValueError):
            predict_from_bundle({"include_task": True}, ["one.png"])


if __name__ == "__main__":
    unittest.main()
