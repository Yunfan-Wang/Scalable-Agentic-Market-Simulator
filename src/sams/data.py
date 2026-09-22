"""Validated bucket-local windows, train-only normalization, and data fingerprints."""

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .schema import FEATURES, TARGETS


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Statistics:
    x_mean: list
    x_std: list
    y_mean: list
    y_std: list

    @classmethod
    def fit(cls, frame):
        train = frame.loc[frame.split == "train"]
        if len(train) < 2:
            raise ValueError("At least two training rows are required for normalization")
        result = []
        for columns, floor in ((FEATURES, 1e-6), (TARGETS, 1e-8)):
            values = train[columns].to_numpy(dtype=np.float32)
            mean, std = values.mean(0), values.std(0, ddof=1)
            result.extend([mean.tolist(), np.where(std > floor, std, 1).tolist()])
        return cls(*result)

    def validate(self):
        for name, size in (("x_mean", 19), ("x_std", 19), ("y_mean", 6), ("y_std", 6)):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid normalization statistic: {name}")
            if name.endswith("std") and not (values > 0).all():
                raise ValueError(f"Standard deviations must be positive: {name}")
        return self

    def to_dict(self):
        return asdict(self)


def validate_frame(frame):
    required = set(FEATURES + TARGETS + ["stock_id", "time_id", "seconds_in_bucket", "split"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing data columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Data must be nonempty and contain no missing values")
    if not np.isfinite(frame[FEATURES + TARGETS].to_numpy(dtype=np.float32)).all():
        raise ValueError("Features and targets must be finite")
    if not set(frame.split).issubset({"train", "validation", "test"}):
        raise ValueError("Unknown data split")
    keys = ["stock_id", "time_id", "seconds_in_bucket"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate stock/bucket/second rows")
    groups = frame.groupby(["stock_id", "time_id"], sort=False)
    if (groups.split.nunique() != 1).any():
        raise ValueError("A bucket cannot span multiple splits")
    if (frame.groupby("time_id").split.nunique() != 1).any():
        raise ValueError("A time_id cannot span multiple splits across stocks")
    for key, bucket in groups:
        seconds = np.sort(bucket.seconds_in_bucket.to_numpy())
        if not np.array_equal(seconds, np.arange(599)):
            raise ValueError(f"Bucket {key} must contain exactly seconds 0..598")
    return frame.sort_values(keys).reset_index(drop=True)


def load_frame(path):
    return validate_frame(pd.read_parquet(path))


class WindowDataset(Dataset):
    """Index windows lazily; never materialize all overlapping sequences."""

    def __init__(self, frame, split, sequence_len, stats):
        stats.validate()
        self.frame = (
            frame.loc[frame.split == split]
            .sort_values(["stock_id", "time_id", "seconds_in_bucket"])
            .reset_index(drop=True)
        )
        if not 2 <= sequence_len <= 599:
            raise ValueError("sequence_len must be between 2 and 599")
        self.sequence_len = sequence_len
        self.x = np.asarray(
            (self.frame[FEATURES].to_numpy(dtype=np.float32) - stats.x_mean) / stats.x_std,
            dtype=np.float32,
        )
        self.y = np.asarray(
            (self.frame[TARGETS].to_numpy(dtype=np.float32) - stats.y_mean) / stats.y_std,
            dtype=np.float32,
        )
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("Normalized data is not finite")
        starts, counts = [], []
        for positions in self.frame.groupby(["stock_id", "time_id"], sort=False).indices.values():
            count = len(positions) - sequence_len + 1
            if count > 0:
                starts.append(positions[0])
                counts.append(count)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.cumulative = np.cumsum(counts)
        if not len(counts):
            raise ValueError(f"No windows available for {split}")

    def __len__(self):
        return int(self.cumulative[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = int(np.searchsorted(self.cumulative, index, side="right"))
        previous = 0 if group == 0 else self.cumulative[group - 1]
        start = self.starts[group] + index - previous
        end = start + self.sequence_len
        return torch.from_numpy(self.x[start:end]), torch.from_numpy(self.y[start:end])
