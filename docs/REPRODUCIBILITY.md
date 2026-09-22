# Reproducibility and limitations

## Research artifacts versus packaged implementation

The original notebook, execution wrapper, catalogue and report are preserved. The package extracts the same core encoder/population/transition architecture while adding explicit configuration, causal initial-value handling, shared time-bucket splitting across stocks, versioned checkpoints and distributed execution.

Consequently, packaged results must be reported separately from historical notebook results. Do not transfer headline accuracy or speedup figures to this implementation without rerunning a documented experiment.

## Run record

Training writes configuration, Python/PyTorch versions, platform, device, process count, data SHA-256, Git revision and dirty status. Checkpoints contain the schema, ordered features/targets, normalization and optimization state. Benchmark JSON includes raw timings, hardware description, dimensions, warmup and seed.

Exact results can vary across library versions and hardware. Seeds and rank-specific random states support controlled reruns; deterministic GPU kernels are not forced. Checkpoint continuation is at epoch boundaries.

## Evidence levels

1. Unit tests establish selected software invariants.
2. Synthetic end-to-end runs establish pipeline operability.
3. Timing experiments measure implementation cost on a named machine.
4. Real-data held-out experiments are needed for predictive claims.
5. Free-rollout and intervention experiments are needed for simulator claims.

The first three do not substitute for the last two.

## Known limitations

- The six-target output does not reconstruct a complete 19-feature next input.
- Learned latent actions are not an exchange execution interface.
- Agent labels in an illustrative demo are not independently validated identities.
- Diagonal Gaussian targets omit cross-target covariance.
- Overlapping evaluation windows do not yield independent samples.
- The deterministic bucket split is not a chronological backtest.
- Data preparation is in memory.
- Frozen-encoder adaptation lacks automated drift checks and rollback.
- Local CPU tests do not certify GPU or multi-node performance.
