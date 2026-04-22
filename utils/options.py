from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yml"


def load_config() -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args, _ = parser.parse_known_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping")
    return config
