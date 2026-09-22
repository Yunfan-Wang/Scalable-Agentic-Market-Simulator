# Data contract

The implemented adapter targets **Optiver Realized Volatility Prediction**, with book_train.parquet and trade_train.parquet directories partitioned by stock_id. The presentation's reference to “Trading at the Close” does not match the executable notebook's book/trade schema. Follow this contract when reproducing the code.

## Features

The authoritative ordered lists are in [schema.py](../src/sams/schema.py).

| Group | Fields |
|---|---|
| Relative book prices | bid_px_rel_1, ask_px_rel_1, bid_px_rel_2, ask_px_rel_2 |
| Log sizes | log_bid_size_1, log_ask_size_1, log_bid_size_2, log_ask_size_2 |
| Market changes | spread, imbalance_1, return_1, spread_change, imbalance_change |
| Trade activity | log_trade_size, log_trade_count |
| Observation timing | real_book_update, trade_occurred, seconds_since_last_real_update, bucket_progress |

Targets are next_return, next_spread_change, next_imbalance_change, next_log_total_depth_change, next_log_trade_size, and next_log_trade_count. These are reduced transitions, not a full future book.

## Preparation guarantees

- Resample each stock/time bucket to one-second observations; forward-fill book state.
- Require an initial observation at second zero. Never fill early rows with future observations.
- Compute next-step targets before dropping the unlabelled final row, yielding seconds 0–598.
- Hash time_id and seed into approximately 70/15/15 train/validation/test splits, shared across stocks.
- Fit means and standard deviations on training rows only.
- Reject missing features, nonfinite values, duplicate seconds, split-crossing buckets and incomplete grids.
- Build overlapping windows inside individual buckets.

The split is deterministic, **not chronological**. A time_id is not assumed to encode temporal order. Chronological generalization needs a separate dataset with reliable timestamps.

Statistics travel with checkpoints. Initialization from a checkpoint retains its normalization, including for frozen-encoder adaptation.

## Memory and scope

Preparation and loading currently materialize the selected dataset in host memory. The window index is lazy, but the parquet reader is not streaming. Start with a subset of stock partitions when resources are limited. Streaming shards and persisted split registries are roadmap work.

Synthetic data exercises the same transform and schema with artificial prices and activity. It is suitable for software tests only.

No market data, credentials, or trained weights are bundled. Acquire data separately and follow its usage terms.
