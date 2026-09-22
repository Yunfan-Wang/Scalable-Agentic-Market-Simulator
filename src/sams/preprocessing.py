"""Notebook feature equations with forward-fill only (no future backfill)."""

import numpy as np
import pandas as pd
from .schema import OPTIVER_BOOK_COLUMNS, OPTIVER_TRADE_COLUMNS

EPS = 1e-12


def validate_optiver_raw_schema(book, trade, stock_id):
    missing_book = OPTIVER_BOOK_COLUMNS - set(book.columns)
    if missing_book:
        raise RuntimeError(
            f"Optiver book schema mismatch for stock_id={stock_id}. Missing columns: "
            + ", ".join(sorted(missing_book))
        )
    if len(trade):
        missing_trade = OPTIVER_TRADE_COLUMNS - set(trade.columns)
        if missing_trade:
            raise RuntimeError(
                f"Optiver trade schema mismatch for stock_id={stock_id}. Missing columns: "
                + ", ".join(sorted(missing_trade))
            )


def resample_bucket_to_one_second(
    book_bucket: pd.DataFrame, trade_bucket: pd.DataFrame, stock_id: int, time_id: int
) -> pd.DataFrame:
    seconds = pd.DataFrame({"seconds_in_bucket": np.arange(600, dtype=np.int16)})
    book_bucket = (
        book_bucket.sort_values("seconds_in_bucket")
        .drop_duplicates(subset=["seconds_in_bucket"], keep="last")
        .copy()
    )
    book_bucket["real_book_update"] = 1.0
    frame = seconds.merge(book_bucket, on="seconds_in_bucket", how="left")
    frame["real_book_update"] = frame["real_book_update"].fillna(0.0).astype(np.float32)
    book_columns = [
        "bid_price1",
        "ask_price1",
        "bid_price2",
        "ask_price2",
        "bid_size1",
        "ask_size1",
        "bid_size2",
        "ask_size2",
    ]
    frame[book_columns] = frame[book_columns].ffill()
    if len(trade_bucket):
        trade_per_second = trade_bucket.groupby("seconds_in_bucket", as_index=False).agg(
            trade_size=("size", "sum"), trade_count=("order_count", "sum")
        )
        frame = frame.merge(trade_per_second, on="seconds_in_bucket", how="left")
    else:
        frame["trade_size"] = 0.0
        frame["trade_count"] = 0.0
    frame["trade_size"] = frame["trade_size"].fillna(0.0)
    frame["trade_count"] = frame["trade_count"].fillna(0.0)
    frame["trade_occurred"] = (frame["trade_count"] > 0).astype(np.float32)
    last_update_second = (
        frame["seconds_in_bucket"].where(frame["real_book_update"] > 0).ffill().fillna(0)
    )
    frame["seconds_since_last_real_update"] = (
        frame["seconds_in_bucket"] - last_update_second
    ).astype(np.float32)
    frame["stock_id"] = int(stock_id)
    frame["time_id"] = int(time_id)
    frame["bucket_progress"] = (frame["seconds_in_bucket"] / 599.0).astype(np.float32)
    mid = (frame["bid_price1"] + frame["ask_price1"]) / 2.0
    frame["mid_price"] = mid
    for level in (1, 2):
        frame[f"bid_px_rel_{level}"] = (frame[f"bid_price{level}"] - mid) / mid.clip(lower=EPS)
        frame[f"ask_px_rel_{level}"] = (frame[f"ask_price{level}"] - mid) / mid.clip(lower=EPS)
        frame[f"log_bid_size_{level}"] = np.log1p(frame[f"bid_size{level}"])
        frame[f"log_ask_size_{level}"] = np.log1p(frame[f"ask_size{level}"])
    frame["spread"] = frame["ask_price1"] - frame["bid_price1"]
    frame["imbalance_1"] = (frame["bid_size1"] - frame["ask_size1"]) / (
        frame["bid_size1"] + frame["ask_size1"]
    ).clip(lower=EPS)
    frame["log_trade_size"] = np.log1p(frame["trade_size"])
    frame["log_trade_count"] = np.log1p(frame["trade_count"])
    frame["log_mid"] = np.log(frame["mid_price"].clip(lower=EPS))
    frame["return_1"] = frame["log_mid"].diff().fillna(0.0)
    frame["spread_change"] = frame["spread"].diff().fillna(0.0)
    frame["imbalance_change"] = frame["imbalance_1"].diff().fillna(0.0)
    frame["total_depth_1"] = (frame["bid_size1"] + frame["ask_size1"]).clip(lower=0.0)
    frame["log_total_depth_1"] = np.log1p(frame["total_depth_1"])
    frame["next_return"] = frame["return_1"].shift(-1)
    frame["next_spread_change"] = frame["spread_change"].shift(-1)
    frame["next_imbalance_change"] = frame["imbalance_change"].shift(-1)
    frame["next_log_total_depth_change"] = (
        frame["log_total_depth_1"].shift(-1) - frame["log_total_depth_1"]
    )
    frame["next_log_trade_size"] = frame["log_trade_size"].shift(-1)
    frame["next_log_trade_count"] = frame["log_trade_count"].shift(-1)
    return frame.iloc[:-1].reset_index(drop=True)
