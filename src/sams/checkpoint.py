"""Versioned, atomic checkpoints containing tensors and plain Python values."""

from pathlib import Path
import os
import torch
from .config import Config
from .data import Statistics
from .schema import SCHEMA_VERSION, FEATURES, TARGETS


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            **payload,
            "format_version": 1,
            "schema": SCHEMA_VERSION,
            "features": FEATURES,
            "targets": TARGETS,
        },
        temporary,
    )
    os.replace(temporary, path)


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("format_version") != 1
        or state.get("schema") != SCHEMA_VERSION
        or state.get("features") != FEATURES
        or state.get("targets") != TARGETS
    ):
        raise ValueError("Checkpoint format or feature/target schema mismatch")
    Config.from_dict(state["config"])
    Statistics(**state["statistics"]).validate()
    return state
