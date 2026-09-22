import numpy as np
import pandas as pd
import pytest
from sams.prepare import transform_bucket, split_time_id


def book_fixture():
    return pd.DataFrame(
        {
            "time_id": [1, 1, 1],
            "seconds_in_bucket": [0, 10, 599],
            "bid_price1": [99.0, 109.0, 119.0],
            "ask_price1": [101.0, 111.0, 121.0],
            "bid_price2": [98.0, 108.0, 118.0],
            "ask_price2": [102.0, 112.0, 122.0],
            "bid_size1": [20.0, 30.0, 40.0],
            "ask_size1": [20.0, 30.0, 40.0],
            "bid_size2": [20.0, 30.0, 40.0],
            "ask_size2": [20.0, 30.0, 40.0],
        }
    )


def test_no_future_backfill():
    book = book_fixture()
    with pytest.raises(ValueError, match="future backfill"):
        transform_bucket(book.iloc[1:], pd.DataFrame(), 0, 1)
    frame = transform_bucket(book, pd.DataFrame(), 0, 1)
    assert (frame.loc[:9, "mid_price"] == 100).all()
    assert frame.loc[10, "mid_price"] == 110
    assert frame.loc[9, "next_return"] == pytest.approx(np.log(110 / 100))
    assert frame.loc[598, "next_return"] == pytest.approx(np.log(120 / 110))
    assert len(frame) == 599


def test_fractional_seconds_rejected():
    book = book_fixture()
    book["seconds_in_bucket"] = book.seconds_in_bucket.astype(float)
    book.loc[1, "seconds_in_bucket"] = 1.5
    with pytest.raises(ValueError, match="integers"):
        transform_bucket(book, pd.DataFrame(), 0, 1)


def test_split_reproducible():
    assert [split_time_id(i) for i in range(100)] == [split_time_id(i) for i in range(100)]
    assert set(split_time_id(i) for i in range(100)) == {"train", "validation", "test"}
