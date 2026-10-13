from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier

from src.tappy_data import build_participant_dataset, build_unlabeled_participant_features
from src.train_tappy_models import predict_participant_features


class TappyDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.event_root = self.root / "Archived-Data" / "Tappy Data"
        self.event_root.mkdir(parents=True)
        with zipfile.ZipFile(self.root / "Archived-users.zip", "w") as archive:
            archive.writestr(
                "Archived users/User_ABCDEFGHIJ.txt",
                "BirthYear: 1960\nParkinsons: False\nGender: Female\n",
            )
            archive.writestr(
                "Archived users/User_KLMNOPQRST.txt",
                "BirthYear: 1955\nParkinsons: True\nGender: Male\n",
            )
            archive.writestr(
                "Archived users/User_UVWXYZABCD.txt",
                "BirthYear: 1970\nParkinsons: False\nGender: Female\n",
            )
        control_rows = [
            "ABCDEFGHIJ\t170101\t10:00:00.000\tL\t100\tLL\t200\t150\t",
            "ABCDEFGHIJ\t170101\t10:00:01.000\tR\t200\tLR\t300\t250\t",
            "ABCDEFGHIJ\t170101\t10:00:02.000\tL\t300\tRL\t400\t350\t",
            "ABCDEFGHIJ\t170101\t10:00:02.250\tS\t100\tSL\t200\t150\t",
            "ABCDEFGHIJ\t170101\t10:00:02.500\tR\t150\tLR\t200\t-50",
            "ABCDEFGHIJ\t170101\t10:00:02.750\tR\t-1\tLR\t200\t150\t",
            "ABCDEFGHIJ\t170101\t10:00:03.000\tR\t150\tLR\t200\t150\t",
            "ABCDEFGHIJ\t170101\t10:00:03.000\tR\t150\tLR\t200\t150\t",
            "ABCDEFGHIJ\t170101\t25:99:99.999\tL\t100\tLL\t200\t150\t",
            "BROKENKEY\t170101\t10:00:03.000\tL\t100\tLL\t200\t150\t",
        ]
        pd_rows = [
            "KLMNOPQRST\t170101\t11:00:00.000\tR\t400\tRR\t500\t450\t",
            "KLMNOPQRST\t170101\t11:00:01.000\tL\t600\tRL\t700\t650\t",
            "KLMNOPQRST\t170101\t11:00:02.000\tR\t500\tLR\t600\t550\t",
        ]
        sparse_rows = ["UVWXYZABCD\t170101\t12:00:00.000\tL\t100\tLL\t200\t150\t"]
        unknown_rows = ["1234567890\t170101\t13:00:00.000\tR\t300\tRR\t400\t350\t"]
        (self.event_root / "ABCDEFGHIJ_1701.txt").write_text("\n".join(control_rows) + "\n", encoding="utf-8")
        (self.event_root / "KLMNOPQRST_1701.txt").write_text("\n".join(pd_rows) + "\n", encoding="utf-8")
        (self.event_root / "UVWXYZABCD_1701.txt").write_text("\n".join(sparse_rows) + "\n", encoding="utf-8")
        (self.event_root / "1234567890_1701.txt").write_text("\n".join(unknown_rows) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_aggregates_once_per_profile_and_excludes_bad_userkey(self) -> None:
        dataset, audit = build_participant_dataset(self.root)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset["target"].value_counts().to_dict(), {0: 2, 1: 1})
        self.assertEqual(audit["matched_event_rows"], 11)
        self.assertEqual(audit["filename_userkey_mismatch_rows"], 1)
        self.assertEqual(audit["invalid_timing_rows"], 1)
        self.assertEqual(audit["duplicate_rows_excluded"], 1)
        self.assertEqual(audit["invalid_timestamp_rows"], 1)
        control = dataset.set_index("user_key").loc["ABCDEFGHIJ"]
        self.assertEqual(control["event_count"], 7)
        self.assertAlmostEqual(control["hold_time_mean_session_median"], 157.1428571429)
        self.assertAlmostEqual(control["flight_time_median_session_median"], 150.0)
        self.assertAlmostEqual(control["hand_s_fraction_session_median"], 1 / 7)
        self.assertEqual(audit["unrecognized_hand_rows"], 0)
        self.assertNotIn("BirthYear", dataset.columns)
        self.assertNotIn("Timestamp", dataset.columns)

    def test_minimum_event_count_is_applied_per_participant(self) -> None:
        dataset, audit = build_participant_dataset(self.root, min_events=2)
        self.assertEqual(audit["excluded_below_min_events"], 1)
        self.assertEqual(len(dataset), 2)

    def test_unlabeled_feature_extraction_keeps_users_without_profiles(self) -> None:
        dataset, _ = build_unlabeled_participant_features(self.root)
        unknown = dataset.set_index("user_key").loc["1234567890"]
        self.assertTrue(unknown["target"] != unknown["target"])  # NaN target
        self.assertEqual(unknown["event_count"], 1)

        feature_columns = [column for column in dataset if column not in {"user_key", "target"}]
        model = DummyClassifier(strategy="prior").fit(dataset[feature_columns], dataset["target"].fillna(0).astype(int))
        model_path = self.root / "model.joblib"
        joblib.dump({"pipeline": model, "feature_columns": feature_columns}, model_path)
        predictions = predict_participant_features(model_path, dataset.drop(columns=["target"]))
        self.assertEqual(len(predictions), len(dataset))
        self.assertIn("decision_score", predictions)
        self.assertIn("user_key", predictions)


if __name__ == "__main__":
    unittest.main()
