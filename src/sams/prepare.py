"""Offline preparation from licensed Optiver book/trade parquet directories."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocessing import resample_bucket_to_one_second, validate_optiver_raw_schema
from .data import validate_frame
from .schema import SCHEMA_VERSION, FEATURES, TARGETS


def split_time_id(time_id, seed=42):
    # Same time bucket is never split across stocks. These IDs are not asserted chronological.
    value = int(hashlib.sha256(f"sams:{seed}:{time_id}".encode()).hexdigest()[:8], 16) / 2**32
    return "train" if value < 0.7 else "validation" if value < 0.85 else "test"


def transform_bucket(book, trade, stock_id, time_id):
    validate_optiver_raw_schema(book, trade, stock_id)
    if book.empty or book.seconds_in_bucket.min() != 0:
        raise ValueError(
            "A book observation at second zero is required; future backfill is forbidden"
        )
    prices = [f"{side}_price{level}" for side in ("bid", "ask") for level in (1, 2)]
    sizes = [f"{side}_size{level}" for side in ("bid", "ask") for level in (1, 2)]
    values = book[prices + sizes].to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (book[prices] <= 0).any().any()
        or (book[sizes] < 0).any().any()
    ):
        raise ValueError("Book prices must be finite and positive; sizes nonnegative")
    for frame in (book, trade):
        if len(frame) and (
            not frame.seconds_in_bucket.between(0, 599).all()
            or not (frame.seconds_in_bucket % 1 == 0).all()
        ):
            raise ValueError("Raw seconds must be integers within 0..599")
    if len(trade):
        trade_values = trade[["size", "order_count"]].to_numpy(dtype=float)
        if not np.isfinite(trade_values).all() or (trade_values < 0).any():
            raise ValueError("Trade activity must be finite and nonnegative")
    return resample_bucket_to_one_second(book, trade, stock_id, time_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-dir", type=Path, required=True)
    parser.add_argument("--trade-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose another path")
    frames = []
    for path in sorted(args.book_dir.glob("stock_id=*")):
        stock = int(path.name.split("=")[1])
        book = pd.read_parquet(path)
        trade_path = args.trade_dir / path.name
        if not trade_path.exists():
            raise FileNotFoundError(f"Missing trade partition: {trade_path}")
        trade = pd.read_parquet(trade_path)
        for time_id, bucket in book.groupby("time_id"):
            frame = transform_bucket(bucket, trade[trade.time_id == time_id], stock, int(time_id))
            frame["split"] = split_time_id(int(time_id), args.seed)
            frames.append(frame)
    if not frames:
        raise ValueError("No stock_id partitions found")
    result = validate_frame(pd.concat(frames, ignore_index=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "rows": len(result),
                "seed": args.seed,
                "features": FEATURES,
                "targets": TARGETS,
                "split_policy": "hash_time_id_across_stocks",
                "future_backfill": False,
                "synthetic": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(result)} rows: {args.output}")


if __name__ == "__main__":
    main()
