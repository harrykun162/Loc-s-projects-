"""
src/utils/config_loader.py
--------------------------
Load configs/config.yaml once and expose a typed settings object.
Also loads .env via python-dotenv so environment variables override
YAML defaults at runtime.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def _load_yaml(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Settings:
    """Dot-access wrapper around the YAML config dict."""

    def __init__(self, data: dict):
        self._data = data
        for k, v in data.items():
            # Only set string keys as attributes (skip numeric keys like in rfm.segments)
            if isinstance(k, str):
                setattr(self, k, Settings(v) if isinstance(v, dict) else v)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Settings({list(self._data.keys())})"


def get_settings() -> Settings:
    cfg = _load_yaml()
    # Override with environment variables where defined
    if os.getenv("APP_PORT"):
        cfg["app"]["port"] = int(os.getenv("APP_PORT"))
    if os.getenv("APP_HOST"):
        cfg["app"]["host"] = os.getenv("APP_HOST")
    if os.getenv("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = os.getenv("MLFLOW_TRACKING_URI")
    if os.getenv("MLFLOW_EXPERIMENT_NAME"):
        cfg["mlflow"]["experiment_name"] = os.getenv("MLFLOW_EXPERIMENT_NAME")
    return Settings(cfg)


# Singleton — import this everywhere
settings = get_settings()
