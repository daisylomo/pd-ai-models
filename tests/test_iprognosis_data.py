"""Regression tests for typing event parsing, linkage, and model evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.iprognosis_data import build_participant_dataset, extract_excerpt_features, parse_event_line
from src.train_typing_models import evaluate, make_participant_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


class IPrognosisParsingTests(unittest.TestCase):
    """Check event validation, feature extraction, and participant linkage."""

    def test_parses_documented_fields_and_millisecond_timestamps(self):
        """Keep parsed timestamps in milliseconds and pressure numeric."""
        self.assertEqual(parse_event_line("Press,1000,Release 1120,0.625"), (1000, 1120, 0.625))

    def test_rejects_malformed_and_reversed_events(self):
        """Reject rows with missing fields or a release before its press."""
        with self.assertRaises(ValueError):
            parse_event_line("Press,1000,1120,0.5")
        with self.assertRaises(ValueError):
            parse_event_line("Press,1200,Release 1100,0.5")

    def test_extracts_hold_interkey_pause_and_pressure_features(self):
        """Summarize event timing and normalized pressure for a sample excerpt."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S01_TEX01.txt"
            path.write_text(
                "Press,1000,Release 1100,0.25\n"
                "Press,1200,Release 1300,0.50\n"
                "Press,2500,Release 2700,0.75\n",
                encoding="utf-8",
            )
            features, audit = extract_excerpt_features(path, pause_threshold_ms=1000)
        self.assertEqual(audit["valid_events"], 3)
        self.assertAlmostEqual(features["hold_ms_median"], 100.0)
        self.assertAlmostEqual(features["press_interval_ms_median"], 750.0)
        self.assertEqual(features["pause_fraction"], 0.5)
        self.assertAlmostEqual(features["pressure_normalized_median"], 0.5)

    def test_links_labels_by_subject_id_and_aggregates_to_one_row_per_participant(self):
        """Join clinical labels by subject ID and median-aggregate excerpts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Data" / "S01").mkdir(parents=True)
            (root / "Data" / "S02").mkdir(parents=True)
            (root / "Data" / "S01" / "S01_TEX01.txt").write_text(
                "Press,100,Release 150,0.25\nPress,250,Release 300,0.5\n", encoding="utf-8"
            )
            (root / "Data" / "S01" / "S01_TEX02.txt").write_text(
                "Press,100,Release 160,0.3\nPress,260,Release 330,0.6\n", encoding="utf-8"
            )
            (root / "Data" / "S02" / "S02_TEX01.txt").write_text(
                "Press,200,Release 260,0.7\nPress,350,Release 430,0.8\n", encoding="utf-8"
            )
            pd.DataFrame({
                "Subject ID": ["01", "02"],
                "Group": ["PD", "Control"],
                "Age (years)": [70, 60],
            }).to_csv(root / "Demographics_Clinical_Characteristics.csv", index=False)
            participants, excerpts, audit = build_participant_dataset(root)
        self.assertEqual(len(participants), 2)
        self.assertEqual(len(excerpts), 3)
        self.assertEqual(participants.set_index("participant_id")["target"].to_dict(), {"01": 1, "02": 0})
        self.assertEqual(audit["participants_by_label"], {"control": 1, "pd": 1})
        self.assertNotIn("Age (years)", participants.columns)
        self.assertEqual(participants.set_index("participant_id").loc["01", "excerpt_count"], 2)


class ParticipantSplitTests(unittest.TestCase):
    """Check participant-level splitting and the shape of metric reports."""

    def test_holdout_is_reproducible_stratified_and_participant_disjoint(self):
        """Use the same seed consistently and never split a person across sets."""
        ids = np.asarray([f"S{i:02d}" for i in range(1, 21)])
        labels = np.asarray([0] * 9 + [1] * 11)
        train, test = make_participant_split(labels, ids, seed=42)
        train_again, test_again = make_participant_split(labels, ids, seed=42)
        self.assertTrue(np.array_equal(train, train_again))
        self.assertTrue(np.array_equal(test, test_again))
        self.assertFalse(set(ids[train]) & set(ids[test]))
        self.assertEqual(set(labels[test]), {0, 1})

    def test_metric_output_includes_balanced_accuracy_and_class_support(self):
        """Include balanced accuracy and both class counts in evaluation output."""
        features = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        labels = np.asarray([0, 0, 1, 1])
        model = Pipeline([("classifier", DummyClassifier(strategy="constant", constant=1))]).fit(features, labels)
        metrics = evaluate(model, features, labels)
        self.assertEqual(metrics["class_support"], {"control": 2, "pd": 2})
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["confusion_matrix_labels_control_pd"], [[0, 2], [0, 2]])


if __name__ == "__main__":
    # Allow this test module to be run directly as well as through unittest discovery.
    unittest.main()
