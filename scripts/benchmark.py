"""Measure the actual packaged model; synthetic inputs are not fidelity evidence."""

import argparse
import json
import platform
import statistics
from datetime import datetime, timezone
from dataclasses import asdict
import time
from pathlib import Path
import torch
from sams.config import ModelConfig
from sams.model import MarketSimulationModel
from sams.data import file_sha256
import sams.model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", type=int, default=[1, 4, 24, 48, 96])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.repeats, args.warmup, *args.agents) < 1:
        parser.error("All dimensions and iteration counts must be positive")
    if Path(args.output).exists():
        parser.error("Output exists")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    x = torch.randn(args.batch_size, args.sequence_length, 19, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    rows = []
    with torch.inference_mode():
        for count in args.agents:
            model = MarketSimulationModel(ModelConfig(num_agents=count)).to(device).eval()
            for _ in range(args.warmup):
                model(x, deterministic=True)
            sync()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                model(x, deterministic=True)
                sync()
                samples.append((time.perf_counter() - start) * 1000)
            rows.append(
                {
                    "agents": count,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "latency_ms": samples,
                    "median_ms": statistics.median(samples),
                }
            )
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_source_sha256": file_sha256(sams.model.__file__),
        "base_model_config": asdict(ModelConfig()),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": str(device),
        "hardware": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
        "seed": 42,
        "threads": 1,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "warmup": args.warmup,
        "mode": "deterministic forward; random weights; synthetic inputs",
        "results": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
