"""Shared input contract and deterministic, duplicate-safe dataset preparation."""
import json
import random
from collections import Counter
from pathlib import Path

LABELS = ["REFLECTION", "RESEARCH", "COLLABORATE", "CREATE", "PRACTICE", "EXPERIENCE", "TEACH"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
FIELDS = ("title", "description", "instructions")


def activity_text(row):
    if not isinstance(row, dict):
        raise ValueError("Each activity must be an object")
    parts = []
    for field in FIELDS:
        value = row.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        parts.append(f"{field}: {' '.join(value.split())}")
    if not any(row.get(field, "").strip() for field in FIELDS):
        raise ValueError("At least one activity text field must be nonempty")
    return "\n".join(parts)


def load_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {number}") from exc
    else:
        rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON array or JSONL file")
    result, seen = [], {}
    for index, row in enumerate(rows, 1):
        text = activity_text(row)
        label = row.get("activityType")
        if not isinstance(label, str) or label not in LABEL2ID:
            raise ValueError(f"Row {index}: activityType must be one of {LABELS}")
        key = text.casefold()
        if key in seen:
            if seen[key] != label:
                raise ValueError(f"Row {index}: duplicate text with conflicting labels")
            continue
        seen[key] = label
        result.append({"text": text, "labels": LABEL2ID[label]})
    return result


def split_records(rows, seed=42):
    """Split each class 70/15/15, retaining at least one in each partition."""
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    for index, label in enumerate(LABELS):
        group = [row for row in rows if row["labels"] == index]
        if len(group) < 3:
            raise ValueError(f"{label}: need at least 3 unique examples; got {len(group)}")
        rng.shuffle(group)
        holdout = max(1, int(len(group) * .15))
        splits["test"].extend(group[:holdout])
        splits["validation"].extend(group[holdout:2 * holdout])
        splits["train"].extend(group[2 * holdout:])
    for split in splits.values():
        rng.shuffle(split)
    return splits


def class_counts(rows):
    counts = Counter(row["labels"] for row in rows)
    return {label: counts[index] for index, label in enumerate(LABELS)}
