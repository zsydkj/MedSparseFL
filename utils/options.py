from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yml"

def load_config() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
