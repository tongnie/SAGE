"""Scenario split loading helpers."""

from __future__ import annotations

import json
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
