"""End-to-end interruption/resume and frozen-encoder adaptation on synthetic data."""

import json
import subprocess
import sys
import torch
import yaml
from sams.checkpoint import load


def run(*args):
    result = subprocess.run(
        [sys.executable, "-m", *map(str, args)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_resume_and_adaptation(tmp_path, frame, config):
    data = tmp_path / "data.parquet"
    # A long window keeps the fixture small while preserving a full valid bucket.
    frame.to_parquet(data, index=False)
    values = config.to_dict()
    values["training"].update(batch_size=128, epochs=2)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    common = [
        "sams.pretrain",
        "--config",
        path,
        "--data",
        data,
        "--device",
        "cpu",
        "--max-steps",
        "1",
    ]
    run(*common, "--output", full)
    run(*common, "--output", resumed, "--stop-after", "1")
    run(*common, "--output", resumed, "--resume", resumed / "latest.pt")
    left, right = load(full / "latest.pt"), load(resumed / "latest.pt")
    assert left["step"] == right["step"] == 2
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key], right["model"][key], rtol=0, atol=0)
    values["training"].update(stage="participants", epochs=1)
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    adapted = tmp_path / "adapted"
    run(*common, "--output", adapted, "--initialize-from", full / "best.pt")
    original, changed = load(full / "best.pt"), load(adapted / "latest.pt")
    for key in original["model"]:
        if key.startswith("encoder."):
            torch.testing.assert_close(
                original["model"][key], changed["model"][key], rtol=0, atol=0
            )
    assert any(
        not torch.equal(original["model"][key], changed["model"][key])
        for key in original["model"]
        if key.startswith("population.")
    )
    result = tmp_path / "test.json"
    run(
        "sams.evaluate",
        "--checkpoint",
        adapted / "best.pt",
        "--data",
        data,
        "--output",
        result,
        "--device",
        "cpu",
    )
    assert json.loads(result.read_text())["tokens"] > 0
