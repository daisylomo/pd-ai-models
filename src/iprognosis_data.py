"""Load and summarize the local i-PROGNOSIS smartphone typing dataset.

The source stores one event per line as ``Press,<ms>,Release <ms>,<pressure>``.
Participant labels come only from the Group column in the clinical CSV. Text
excerpt counts and event counts are audit fields, not model predictors.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IPROGNOSIS_ROOT = (
    PROJECT_ROOT / "data" / "Parkinson's Datasets" / "virtual keyboard data" / "iprognosis_typing_dev"
)
LABELS = {"Control": 0, "PD": 1}
EVENT_FILE_RE = re.compile(r"^S(\d{2})_TEX(\d{1,3})\.txt$", re.IGNORECASE)


def parse_event_line(line: str) -> tuple[int, int, float]:
    """Parse the documented four-field event row; timestamps are milliseconds."""
    fields = [field.strip() for field in line.strip().split(",")]
    if len(fields) != 4:
        raise ValueError("expected four comma-separated fields")
    if fields[0] != "Press" or not fields[2].startswith("Release "):
        raise ValueError("expected 'Press' and 'Release' field markers")
    try:
        press_ms = int(fields[1])
        release_ms = int(fields[2][len("Release "):].strip())
        pressure = float(fields[3])
    except ValueError as error:
        raise ValueError("timestamp or pressure is not numeric") from error
    if not np.isfinite(pressure):
        raise ValueError("pressure must be finite")
    if press_ms < 0 or release_ms < 0:
        raise ValueError("timestamps must be non-negative")
    if release_ms < press_ms:
        raise ValueError("release timestamp precedes press timestamp")
    return press_ms, release_ms, pressure


def _summaries(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    q25, median, q75, q90 = (float(v) for v in np.quantile(array, [0.25, 0.5, 0.75, 0.9]))
    std = float(array.std(ddof=0))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_median": median,
        f"{prefix}_q90": q90,
        f"{prefix}_iqr": q75 - q25,
        f"{prefix}_cv": std / mean if mean > 0 else 0.0,
    }


def extract_excerpt_features(path: Path, *, pause_threshold_ms: int = 1000) -> tuple[dict[str, float], dict[str, Any]]:
    """Extract timing and in-range normalized-pressure summaries for one excerpt."""
    events: list[tuple[int, int, float]] = []
    malformed_lines = 0
    invalid_events = 0
    out_of_range_pressure = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                event = parse_event_line(line)
            except ValueError as error:
                if "release timestamp precedes" in str(error):
                    invalid_events += 1
                else:
                    malformed_lines += 1
                continue
            if not 0.0 <= event[2] <= 1.0:
                out_of_range_pressure += 1
            events.append(event)

    holds: list[float] = []
    press_intervals: list[float] = []
    interkey_gaps: list[float] = []
    pauses: list[float] = []
    pressures: list[float] = []
    timestamp_reversals = 0
    invalid_intervals = 0
    for press, release, pressure in events:
        holds.append(float(release - press))
        # The source defines pressure as normalized 0-1. Retain anomalous raw
        # values in the audit, but only summarize values within that range.
        if 0.0 <= pressure <= 1.0:
            pressures.append(pressure)
    for previous, current in zip(events, events[1:]):
        press_interval = current[0] - previous[0]
        gap = current[0] - previous[1]
        if press_interval < 0:
            timestamp_reversals += 1
        elif press_interval > 0:
            press_intervals.append(float(press_interval))
        if gap < 0:
            invalid_intervals += 1
        else:
            interkey_gaps.append(float(gap))
            if gap >= pause_threshold_ms:
                pauses.append(float(gap))

    features: dict[str, float] = {}
    features.update(_summaries(holds, "hold_ms"))
    features.update(_summaries(press_intervals, "press_interval_ms"))
    features.update(_summaries(interkey_gaps, "interkey_gap_ms"))
    features.update(_summaries(pauses, "pause_ms"))
    features.update(_summaries(pressures, "pressure_normalized"))
    n_valid_gaps = len(interkey_gaps)
    features["pause_fraction"] = len(pauses) / n_valid_gaps if n_valid_gaps else 0.0
    audit = {
        "nonblank_lines": len(events) + malformed_lines + invalid_events,
        "valid_events": len(events),
        "malformed_lines": malformed_lines,
        "invalid_events": invalid_events,
        "out_of_range_pressure_events": out_of_range_pressure,
        "timestamp_order_reversals": timestamp_reversals,
        "negative_interkey_gaps_excluded": invalid_intervals,
        "pause_threshold_ms": int(pause_threshold_ms),
    }
    return features, audit


def build_participant_dataset(
    iprognosis_root: Path = DEFAULT_IPROGNOSIS_ROOT,
    *,
    pause_threshold_ms: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return participant-level features, excerpt features and a data audit."""
    root = Path(iprognosis_root)
    data_root = root / "Data"
    label_path = root / "Demographics_Clinical_Characteristics.csv"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Typing data directory not found: {data_root}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Clinical label CSV not found: {label_path}")

    labels = pd.read_csv(label_path, dtype={"Subject ID": str})
    required = {"Subject ID", "Group"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Clinical CSV is missing required columns: {sorted(required - set(labels.columns))}")
    labels["participant_id"] = labels["Subject ID"].str.strip().str.zfill(2)
    if labels["participant_id"].duplicated().any():
        raise ValueError("Clinical CSV contains duplicate subject IDs")
    unknown_labels = sorted(set(labels["Group"].dropna()) - set(LABELS))
    if unknown_labels or labels["Group"].isna().any():
        raise ValueError(f"Unexpected or missing diagnostic labels: {unknown_labels}")
    labels["target"] = labels["Group"].map(LABELS).astype(int)

    label_map = labels.set_index("participant_id")["target"].to_dict()
    paths = sorted(data_root.glob("S*/*.txt"))
    if not paths:
        raise FileNotFoundError(f"No excerpt text files found under {data_root}")
    seen_subjects: set[str] = set()
    excerpt_rows: list[dict[str, Any]] = []
    audit_counts: Counter[str] = Counter()
    malformed_files: list[str] = []
    empty_files: list[str] = []

    for path in paths:
        match = EVENT_FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected excerpt filename: {path}")
        participant_id = match.group(1)
        if path.parent.name.upper() != f"S{participant_id}":
            raise ValueError(f"Participant ID disagrees between folder and filename: {path}")
        if participant_id not in label_map:
            raise ValueError(f"No clinical label for participant {participant_id}")
        seen_subjects.add(participant_id)
        features, file_audit = extract_excerpt_features(path, pause_threshold_ms=pause_threshold_ms)
        audit_counts.update({
            key: int(value) for key, value in file_audit.items()
            if isinstance(value, int) and key != "pause_threshold_ms"
        })
        if file_audit["malformed_lines"]:
            malformed_files.append(str(path))
        if file_audit["valid_events"] == 0:
            empty_files.append(str(path))
        excerpt_rows.append({
            "participant_id": participant_id,
            "excerpt_id": int(match.group(2)),
            "target": int(label_map[participant_id]),
            "event_count_audit_only": int(file_audit["valid_events"]),
            **features,
        })

    missing_label_subjects = sorted(set(label_map) - seen_subjects)
    if missing_label_subjects:
        raise ValueError(f"Clinical subjects have no excerpt files: {missing_label_subjects}")
    excerpts = pd.DataFrame(excerpt_rows).sort_values(["participant_id", "excerpt_id"]).reset_index(drop=True)
    feature_columns = [
        column for column in excerpts.columns
        if column not in {"participant_id", "excerpt_id", "target", "event_count_audit_only"}
    ]
    participants = excerpts.groupby(["participant_id", "target"], as_index=False)[feature_columns].median()
    counts = excerpts.groupby("participant_id").agg(
        excerpt_count=("excerpt_id", "size"), total_valid_events=("event_count_audit_only", "sum")
    ).reset_index()
    participants = participants.merge(counts, on="participant_id", validate="one_to_one")

    audit = {
        "dataset": "i-PROGNOSIS touchscreen typing development dataset",
        "label_source": "Group column in Demographics_Clinical_Characteristics.csv; Control=0, PD=1",
        "participant_id_source": "S## folder and S##_TEX## filename linked to Subject ID column",
        "event_schema": ["Press", "press timestamp milliseconds", "Release timestamp milliseconds", "normalized pressure 0-1"],
        "timestamp_unit": "milliseconds, as documented in local ReadMe.txt",
        "pressure_summary_policy": "Only observed values in documented [0,1] range enter pressure features; all values are retained in audit counts, no clipping",
        "aggregation": "median of excerpt-level feature summaries per participant; event and excerpt counts are audit-only",
        "participants": int(len(participants)),
        "participants_by_label": {"control": int((participants.target == 0).sum()), "pd": int((participants.target == 1).sum())},
        "excerpt_files": int(len(excerpts)),
        "excerpt_files_by_participant_min": int(counts.excerpt_count.min()),
        "excerpt_files_by_participant_max": int(counts.excerpt_count.max()),
        "audit_totals": dict(audit_counts),
        "malformed_files": malformed_files,
        "empty_files": empty_files,
        "licence": "Creative Commons Attribution 4.0 International, documented in local ReadMe.txt",
        "required_citation": "Iakovakis et al. (2018), Scientific Reports 8, 7663, doi:10.1038/s41598-018-25999-0",
        "source_readme_notes": "ReadMe reports 10 participants did not complete the shared initial 200-character excerpt; excerpt text/key content is not included, so completeness cannot be independently verified from event files alone.",
    }
    return participants.sort_values("participant_id").reset_index(drop=True), excerpts, audit
