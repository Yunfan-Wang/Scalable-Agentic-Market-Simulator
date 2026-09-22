"""Portable local two-process CPU training with a file rendezvous."""

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time


def worker(rank, rendezvous, arguments):
    os.environ.update(
        WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank), SAMS_INIT_METHOD=rendezvous
    )
    sys.argv = ["sams.pretrain", *arguments]
    from sams.pretrain import main

    main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("timeout must be positive")
    config = Path(__file__).resolve().parents[1] / "configs/smoke.yaml"
    arguments = [
        "--config",
        str(config),
        "--data",
        str(Path(args.data).resolve()),
        "--output",
        str(Path(args.output).resolve()),
        "--device",
        "cpu",
        "--max-steps",
        "2",
    ]
    with tempfile.TemporaryDirectory(prefix="sams-ddp-") as directory:
        rendezvous = (Path(directory) / "store").as_uri()
        context = mp.get_context("spawn")
        processes = [
            context.Process(target=worker, args=(rank, rendezvous, arguments)) for rank in range(2)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + args.timeout
        try:
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            if any(p.exitcode != 0 for p in processes):
                raise RuntimeError(
                    f"Distributed smoke failed or timed out: {[p.exitcode for p in processes]}"
                )
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()


if __name__ == "__main__":
    main()
