"""Scenario split loading helpers."""

from __future__ import annotations

import json
import csv
from pathlib import Path


def load_split_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario split file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_ids(path: str | Path, split: str = "train", max_scenarios: int | None = None) -> list[int]:
    config = load_split_config(path)
    if split not in config["splits"]:
        valid = ", ".join(sorted(config["splits"]))
        raise KeyError(f"Unknown split '{split}'. Valid splits: {valid}")
    split_cfg = config["splits"][split]
    if "ids" in split_cfg:
        ids = list(split_cfg["ids"])
    else:
        start, end = split_cfg["range"]
        ids = list(range(start, end))
    skip_ids = set(config.get("skip_ids", [])) | set(split_cfg.get("skip_ids", []))
    ids = [int(sid) for sid in ids if int(sid) not in skip_ids]
    if max_scenarios is not None:
        ids = ids[:max_scenarios]
    return ids


def scenario_ids_from_csv(path: str | Path) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario summary CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Scenario summary CSV has no header: {path}")
        id_column = next((name for name in ("scenario_id", "seed_id", "id") if name in reader.fieldnames), None)
        if id_column is None:
            raise KeyError(
                f"Scenario summary CSV must contain one of scenario_id, seed_id, or id columns: {path}"
            )
        ids: list[int] = []
        for row in reader:
            value = row.get(id_column, "")
            if value == "":
                continue
            ids.append(int(float(value)))
    return ids


def filter_ids_by_summary(ids: list[int], summary_csv: str | Path | None) -> list[int]:
    if summary_csv is None:
        return ids
    summary_ids = set(scenario_ids_from_csv(summary_csv))
    return [sid for sid in ids if sid in summary_ids]
