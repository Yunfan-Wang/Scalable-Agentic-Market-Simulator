import numpy as np
import pytest
from sams.data import Statistics, WindowDataset, validate_frame
from sams.schema import FEATURES
from sams.distributed import EvaluationSampler


def test_train_only_statistics(frame):
    stats = Statistics.fit(frame)
    changed = frame.copy()
    changed.loc[changed.split != "train", FEATURES] += 10000
    assert stats == Statistics.fit(changed)


def test_windows_stay_in_bucket(frame):
    stats = Statistics.fit(frame)
    dataset = WindowDataset(frame, "train", 129, stats)
    assert len(dataset) == 471
    x, y = dataset[len(dataset) - 1]
    assert x.shape == (129, 19) and y.shape == (129, 6)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


@pytest.mark.parametrize("defect", ["duplicate", "gap", "split", "infinite"])
def test_rejects_bad_frames(frame, defect):
    changed = frame.copy()
    if defect == "duplicate":
        changed.loc[1, "seconds_in_bucket"] = 0
    elif defect == "gap":
        changed = changed.drop(index=1)
    elif defect == "split":
        changed.loc[0, "split"] = "test"
    else:
        changed.loc[0, FEATURES[0]] = np.inf
    with pytest.raises(ValueError):
        validate_frame(changed)


def test_evaluation_has_no_duplicate_padding():
    shards = [list(EvaluationSampler(range(7), rank, 3)) for rank in range(3)]
    assert sorted(sum(shards, [])) == list(range(7))


def test_same_time_across_stocks_cannot_leak(frame):
    changed = frame.copy()
    changed.loc[changed.split == "test", "time_id"] = changed.loc[
        changed.split == "train", "time_id"
    ].iloc[0]
    changed.loc[changed.split == "test", "stock_id"] = 99
    with pytest.raises(ValueError, match="across stocks"):
        validate_frame(changed)
