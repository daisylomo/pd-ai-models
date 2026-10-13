"""Load, audit and aggregate the PhysioNet Tappy Keystroke data.

Each event file is one participant-month recording. Feature summaries are first
computed per recording and then combined with an equal-weight median per person.
This avoids allowing a participant's longest month to dominate their feature
row. Usage counts are retained for auditing but are never model predictors.
"""

from __future__ import annotations

import csv
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAPPY_ROOT = PROJECT_ROOT / "data" / "Parkinson's Datasets" / "tappy-keystroke-data-1.0.0"
TIMING_COLUMNS = ("Hold time", "Latency time", "Flight time")
EXPECTED_FIELDS = 8
DIRECTION_LEVELS = ("LL", "LR", "RL", "RR", "LS", "SL", "RS", "SR", "SS")
SESSION_STATISTICS = ("mean", "std", "median", "q90", "iqr", "cv")


def load_profiles(tappy_root: Path = DEFAULT_TAPPY_ROOT) -> dict[str, dict[str, str]]:
    """Read participant profile text files directly from Archived-users.zip."""
    archive = tappy_root / "Archived-users.zip"
    profiles: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(archive) as source:
        for member in source.namelist():
            match = re.search(r"User_([A-Z0-9]{10})\.txt$", member, flags=re.IGNORECASE)
            if not match:
                continue
            profile: dict[str, str] = {}
            for line in source.read(member).decode("utf-8-sig", errors="replace").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    profile[key.strip()] = value.strip()
            profiles[match.group(1).upper()] = profile
    return profiles


def _session_statistics(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std())
    q25, median, q75, q90 = (float(value) for value in np.quantile(array, [0.25, 0.5, 0.75, 0.9]))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": median,
        f"{prefix}_q90": q90,
        f"{prefix}_iqr": q75 - q25,
        f"{prefix}_cv": std / mean if mean > 0 else 0.0,
    }


def _session_features(
    timing_values: dict[str, list[float]],
    hand_values: dict[str, dict[str, list[float]]],
    hand_counts: Counter[str],
    direction_counts: Counter[str],
    event_count: int,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for timing, values in timing_values.items():
        prefix = timing.lower().replace(" ", "_")
        features.update(_session_statistics(values, prefix))
        for hand in ("L", "R"):
            features.update(_session_statistics(hand_values[hand][timing], f"{prefix}_{hand.lower()}"))
    for hand in ("L", "R"):
        features[f"hand_{hand.lower()}_fraction"] = hand_counts[hand] / event_count
    features["hand_s_fraction"] = hand_counts["S"] / event_count
    known_directions = sum(direction_counts.values())
    denominator = max(known_directions, 1)
    for direction in DIRECTION_LEVELS:
        features[f"direction_{direction.lower()}_fraction"] = direction_counts[direction] / denominator
    return features


def _combine_sessions(user_key: str, target: int | None, bucket: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "user_key": user_key,
        "target": target,
        # Audit-only exposure fields. train_tappy_models explicitly excludes them.
        "event_count": int(bucket["event_count"]),
        "session_count": len(bucket["sessions"]),
    }
    names = sorted({name for session in bucket["sessions"] for name in session})
    for name in names:
        values = [session[name] for session in bucket["sessions"] if name in session]
        row[f"{name}_session_median"] = float(np.median(values))
    return row


def _profile_audit(users: dict[str, dict[str, Any]], profiles: dict[str, dict[str, str]]) -> dict[str, Any]:
    fields = ("BirthYear", "Gender", "Tremors", "DiagnosisYear", "Sided", "UPDRS", "Impact", "Levadopa", "DA", "MAOB", "Other")
    result: dict[str, Any] = {}
    for field in fields:
        observed: dict[str, Counter[str]] = {"control": Counter(), "parkinson": Counter()}
        missing = {"control": 0, "parkinson": 0}
        for user_key, bucket in users.items():
            label = "parkinson" if bucket["target"] == 1 else "control"
            value = profiles.get(user_key, {}).get(field, "").strip()
            if not value or value in {"------", "Don't know"}:
                missing[label] += 1
            else:
                observed[label][value] += 1
        result[field] = {
            "missing_by_label": missing,
            "observed_counts_by_label": {label: dict(counts) for label, counts in observed.items()},
        }
    return result


def _valid_date(date_text: str, cache: dict[str, bool]) -> bool:
    if date_text not in cache:
        try:
            cache[date_text] = len(date_text) == 6 and date_text.isdigit() and bool(datetime.strptime(date_text, "%y%m%d"))
        except ValueError:
            cache[date_text] = False
    return cache[date_text]


def _valid_time(time_text: str) -> bool:
    if not re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{3}", time_text):
        return False
    hour, minute, second = int(time_text[0:2]), int(time_text[3:5]), int(time_text[6:8])
    return hour < 24 and minute < 60 and second < 60


def build_participant_dataset(
    tappy_root: Path = DEFAULT_TAPPY_ROOT,
    *,
    min_events: int = 0,
    include_unlabeled: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return participant-level features plus a reproducible data-quality audit.

    Date/time, participant identifiers and profile values are never features.
    Timestamp anomalies are counted, not used to exclude otherwise valid timing
    observations. Exact duplicate rows within a monthly file are excluded and
    counted. Negative flight values are allowed; negative hold/latency are not.
    """
    if min_events < 0:
        raise ValueError("min_events must be non-negative")
    profiles = load_profiles(tappy_root)
    event_root = tappy_root / "Archived-Data" / "Tappy Data"
    if not event_root.is_dir():
        raise FileNotFoundError(f"Tappy event directory not found: {event_root}")

    users: dict[str, dict[str, Any]] = {}
    audit: dict[str, Any] = {
        "event_files": 0,
        "profiles": len(profiles),
        "matched_event_rows": 0,
        "unmatched_profile_event_files": 0,
        "unmatched_profile_event_rows": 0,
        "malformed_rows": 0,
        "filename_userkey_mismatch_rows": 0,
        "duplicate_rows_excluded": 0,
        "invalid_timing_rows": 0,
        "unrecognized_hand_rows": 0,
        "unknown_direction_rows": 0,
        "invalid_date_rows": 0,
        "invalid_timestamp_rows": 0,
        "date_month_filename_mismatches": 0,
        "timestamp_order_reversals": 0,
        "excluded_below_min_events": 0,
        "min_events": int(min_events),
    }
    date_cache: dict[str, bool] = {}

    for path in sorted(event_root.glob("*.txt")):
        audit["event_files"] += 1
        match = re.match(r"^([A-Z0-9]{10})_(\d{4})$", path.stem, flags=re.IGNORECASE)
        if not match:
            audit["malformed_rows"] += 1
            continue
        user_key, file_month = match.group(1).upper(), match.group(2)
        profile = profiles.get(user_key)
        if profile is None:
            audit["unmatched_profile_event_files"] += 1
            if not include_unlabeled:
                with path.open("r", encoding="utf-8-sig", errors="replace") as source:
                    audit["unmatched_profile_event_rows"] += sum(bool(line.strip()) for line in source)
                continue
            with path.open("r", encoding="utf-8-sig", errors="replace") as source:
                audit["unmatched_profile_event_rows"] += sum(bool(line.strip()) for line in source)
            target = None
        else:
            raw_label = profile.get("Parkinsons", "").strip().lower()
            target = int(raw_label == "true") if raw_label in {"true", "false"} else None
            if target is None and not include_unlabeled:
                continue

        bucket = users.setdefault(user_key, {"target": target, "event_count": 0, "sessions": []})
        if bucket["target"] is not None and target is not None and bucket["target"] != target:
            raise ValueError(f"Conflicting profile labels for participant {user_key}")
        if bucket["target"] is None and target is not None:
            bucket["target"] = target

        timing_values = {name: [] for name in TIMING_COLUMNS}
        hand_values = {hand: {name: [] for name in TIMING_COLUMNS} for hand in ("L", "R")}
        hand_counts: Counter[str] = Counter()
        direction_counts: Counter[str] = Counter()
        seen_rows: set[tuple[str, ...]] = set()
        previous_stamp: str | None = None

        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source:
            sample = source.read(4096)
            source.seek(0)
            delimiter = "\t" if "\t" in sample else ","
            reader = csv.reader(source, delimiter=delimiter)
            for values in reader:
                if not values or not any(value.strip() for value in values):
                    continue
                if len(values) not in {EXPECTED_FIELDS, EXPECTED_FIELDS + 1} or (
                    len(values) == EXPECTED_FIELDS + 1 and values[-1].strip()
                ):
                    audit["malformed_rows"] += 1
                    continue
                row = tuple(value.strip() for value in values[:EXPECTED_FIELDS])
                if row in seen_rows:
                    audit["duplicate_rows_excluded"] += 1
                    continue
                seen_rows.add(row)
                if row[0].upper() != user_key:
                    audit["filename_userkey_mismatch_rows"] += 1
                    continue

                date_text, time_text = row[1], row[2]
                date_ok = _valid_date(date_text, date_cache)
                time_ok = _valid_time(time_text)
                if not date_ok:
                    audit["invalid_date_rows"] += 1
                if not time_ok:
                    audit["invalid_timestamp_rows"] += 1
                if date_ok and date_text[:4] != file_month:
                    audit["date_month_filename_mismatches"] += 1
                if date_ok and time_ok:
                    stamp = f"{date_text} {time_text}"
                    if previous_stamp is not None and stamp < previous_stamp:
                        audit["timestamp_order_reversals"] += 1
                    previous_stamp = stamp

                try:
                    timings = {
                        "Hold time": float(row[4]),
                        "Latency time": float(row[6]),
                        "Flight time": float(row[7]),
                    }
                except ValueError:
                    audit["invalid_timing_rows"] += 1
                    continue
                if not all(np.isfinite(value) for value in timings.values()) or (
                    timings["Hold time"] < 0 or timings["Latency time"] < 0
                ):
                    audit["invalid_timing_rows"] += 1
                    continue
                hand, direction = row[3].upper(), row[5].upper()
                for name, value in timings.items():
                    timing_values[name].append(value)
                    if hand in hand_values:
                        hand_values[hand][name].append(value)
                if hand in {"L", "R", "S"}:
                    hand_counts[hand] += 1
                else:
                    audit["unrecognized_hand_rows"] += 1
                if direction in DIRECTION_LEVELS:
                    direction_counts[direction] += 1
                else:
                    audit["unknown_direction_rows"] += 1

        n_session = len(timing_values["Hold time"])
        if n_session:
            audit["matched_event_rows"] += n_session
            bucket["event_count"] += n_session
            bucket["sessions"].append(
                _session_features(timing_values, hand_values, hand_counts, direction_counts, n_session)
            )

    rows = []
    included_users: dict[str, dict[str, Any]] = {}
    for user_key, bucket in sorted(users.items()):
        if bucket["event_count"] < min_events:
            audit["excluded_below_min_events"] += 1
            continue
        rows.append(_combine_sessions(user_key, bucket["target"], bucket))
        included_users[user_key] = bucket
    dataset = pd.DataFrame(rows)
    if dataset.empty or (not include_unlabeled and dataset["target"].nunique() != 2):
        raise ValueError("Matched Tappy cohort must contain participants from both label classes")

    audit["participants"] = int(len(dataset))
    audit["participants_by_label"] = {
        "control": int((dataset["target"] == 0).sum()),
        "parkinson": int((dataset["target"] == 1).sum()),
    }
    audit["unlabeled_participants"] = int(dataset["target"].isna().sum())
    audit["sessions_by_label"] = {
        label: int(sum(len(included_users[key]["sessions"]) for key in included_users
                       if included_users[key]["target"] == target))
        for label, target in (("control", 0), ("parkinson", 1), ("unlabeled", None))
    }
    audit["event_count_quantiles_by_label"] = {}
    for label, target in (("control", 0), ("parkinson", 1), ("unlabeled", None)):
        values = dataset.loc[dataset["target"].isna() if target is None else dataset["target"].eq(target), "event_count"]
        if len(values):
            audit["event_count_quantiles_by_label"][label] = {
                "min": int(values.min()), "median": float(values.median()), "max": int(values.max())
            }
    audit["profile_confounds"] = _profile_audit(included_users, profiles)
    audit["feature_columns"] = [
        column for column in dataset.columns if column not in {"user_key", "target", "event_count", "session_count"}
    ]
    audit["missing_feature_values"] = {
        column: int(dataset[column].isna().sum())
        for column in audit["feature_columns"] if dataset[column].isna().any()
    }
    audit["recording_unit"] = "participant-month text file; all participant-month files remain aggregated within one participant row"
    audit["user_key_source"] = "ten-character file-name prefix verified against event UserKey and profile filename"
    audit["label_source"] = "self-reported Parkinsons field from Archived-users.zip"
    return dataset, audit


def build_unlabeled_participant_features(
    tappy_root: Path = DEFAULT_TAPPY_ROOT,
    *,
    min_events: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract participant features for inference, including IDs without profiles."""
    return build_participant_dataset(tappy_root, min_events=min_events, include_unlabeled=True)
