# Contributing

Start with a reproducible issue: the command, configuration, expected behavior, observed behavior and a synthetic example where possible. Never attach credentials or restricted market data.

For changes:

1. Install with python -m pip install -e ".[dev]".
2. Keep model and data-contract changes explicit; update schema/checkpoint handling when semantics change.
3. Run python -m ruff check src scripts tests and python -m pytest -q.
4. Include evidence for correctness and label synthetic, historical and real-data measurements separately.
5. Keep the original research notebook intact unless the change specifically concerns it.

Avoid performance claims without raw measurements and environment details. The repository currently has no selected software license; clarify reuse terms before distributing derivatives.
