"""Ordered Optiver Realized Volatility feature contract."""

SCHEMA_VERSION = "optiver_1s_six_target_v2"
FEATURES = [
    "bid_px_rel_1",
    "ask_px_rel_1",
    "log_bid_size_1",
    "log_ask_size_1",
    "bid_px_rel_2",
    "ask_px_rel_2",
    "log_bid_size_2",
    "log_ask_size_2",
    "spread",
    "imbalance_1",
    "return_1",
    "spread_change",
    "imbalance_change",
    "log_trade_size",
    "log_trade_count",
    "real_book_update",
    "trade_occurred",
    "seconds_since_last_real_update",
    "bucket_progress",
]

TARGETS = [
    "next_return",
    "next_spread_change",
    "next_imbalance_change",
    "next_log_total_depth_change",
    "next_log_trade_size",
    "next_log_trade_count",
]

OPTIVER_BOOK_COLUMNS = {
    "time_id",
    "seconds_in_bucket",
    "bid_price1",
    "ask_price1",
    "bid_price2",
    "ask_price2",
    "bid_size1",
    "ask_size1",
    "bid_size2",
    "ask_size2",
}

OPTIVER_TRADE_COLUMNS = {"time_id", "seconds_in_bucket", "size", "order_count"}
