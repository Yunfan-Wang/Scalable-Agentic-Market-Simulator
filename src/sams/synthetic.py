"""Small seeded synthetic books for installation tests, never market evidence."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .prepare import transform_bucket
from .data import validate_frame


def synthetic_frame(seed=42, buckets_per_split=1):
    if buckets_per_split < 1:
        raise ValueError("buckets_per_split must be positive")
    rng = np.random.default_rng(seed)
    frames = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for bucket in range(buckets_per_split):
            time_id = split_index * buckets_per_split + bucket
            mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.0001, 600)))
            spread = rng.uniform(0.01, 0.03, 600)
            book = pd.DataFrame({"time_id": time_id, "seconds_in_bucket": np.arange(600)})
            for level in (1, 2):
                book[f"bid_price{level}"] = mid - spread * level / 2
                book[f"ask_price{level}"] = mid + spread * level / 2
                book[f"bid_size{level}"] = rng.integers(1, 100, 600)
                book[f"ask_size{level}"] = rng.integers(1, 100, 600)
            trade = pd.DataFrame(
                {
                    "time_id": time_id,
                    "seconds_in_bucket": np.arange(600),
                    "size": rng.integers(0, 20, 600),
                    "order_count": rng.integers(0, 5, 600),
                }
            )
            frame = transform_bucket(book, trade, 0, time_id)
            frame["split"] = split
            frames.append(frame)
    return validate_frame(pd.concat(frames, ignore_index=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buckets-per-split", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose another path")
    frame = synthetic_frame(args.seed, args.buckets_per_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "seed": args.seed,
                "rows": len(frame),
                "purpose": "software smoke testing only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Synthetic fixture: {args.output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
