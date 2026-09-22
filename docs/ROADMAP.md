# Research roadmap

Items below are planned unless explicitly checked. They are acceptance criteria, not release dates.

## Reproducible core

- [x] Installable model and strict configuration
- [x] Offline data contract and train-only statistics
- [x] Joint training, checkpoints and epoch resume
- [x] Encoder-frozen adaptation primitive
- [x] Held-out evaluation and actual two-process distributed test
- [x] Packaged-model latency benchmark with raw timing output

## Evidence before richer simulation

- [ ] Compare persistence, linear, Transformer-only and parameter-matched non-agent baselines on identical splits.
- [ ] Run agent-count and action-bottleneck ablations across multiple seeds.
- [ ] Measure interval coverage, calibration and tail errors for each target.
- [ ] Publish hardware, commit, data fingerprint, split, complete commands and uncertainty for every reported result.
- [ ] Evaluate out-of-period and out-of-instrument generalization on datasets with appropriate timestamps.

## Close the feedback loop

- [ ] Reconstruct all 19 features from a coherent generated state and observation process.
- [ ] Enforce positive depths, valid spreads and consistent trade/book timing.
- [ ] Test recurrent state handling without replaying previously consumed history.
- [ ] Compare free rollouts at increasing horizons against observed distributional statistics.
- [ ] Evaluate volatility clustering, return tails, spread/depth distributions and impact responses.

## Adaptation and scale

- [ ] Add bounded participant updates with acceptance checks, rollback and drift detection.
- [ ] Add streamed parquet shards and restartable preprocessing.
- [ ] Align external event features causally before enabling external-channel training.
- [ ] Validate GPU and multi-node throughput with scaling efficiency and communication breakdowns.
- [ ] Investigate agent-parallel partitioning only after profiling identifies a population bottleneck.

## Release readiness

- [ ] Select a software license.
- [ ] Publish reproducible trained weights with data provenance and model limitations.
- [ ] Version the dataset contract and checkpoint migrations.
- [ ] Add documented release tags and a changelog tied to measured changes.
