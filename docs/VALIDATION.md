# Local validation record

Validated on Windows 11 with Python 3.12.14 and PyTorch 2.14.0+cpu, September 22, 2026.

| Check | Outcome |
|---|---|
| Ruff on src, scripts and tests | Passed |
| Pytest suite | 19 passed |
| Two-process Gloo synchronization and collective failure propagation | Passed |
| Synthetic joint training and test evaluation | Passed |
| Interrupted epoch training versus uninterrupted training | Identical model tensors after resume |
| Encoder-frozen adaptation | Encoder unchanged; population updated |
| Two-process full training via file rendezvous, followed by evaluation | Passed |
| Wheel and source distribution build | Passed |
| Local Markdown link resolution | Passed |
| CPU inference scaling benchmark and chart generation | Completed; raw timings checked in |

The suite also covers causal attention, independent agent state, finite gradients, train-only normalization, bucket-local windows, cross-stock split leakage, future-backfill prevention, next-step alignment, checkpoint schema rejection and unpadded evaluation shards.

The Transformer emits a warning that nested-tensor optimization is disabled with norm_first=True. This is expected for the retained reference architecture.

Standalone torchrun TCP rendezvous failed on this Windows build because PyTorch lacks libuv support, including with USE_LIBUV=0. The file-based launcher exercised the same distributed trainer successfully. Linux CI is configured but has not been run remotely as part of this local change.

No real-data retraining, GPU validation, multi-node benchmark or autonomous rollout-fidelity validation was performed. Synthetic metrics are software checks and are not reported as predictive research results.
