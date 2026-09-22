# Training and adaptation

Install the editable package and prepare data using the README commands. “Pretraining” here means initial joint fitting of the encoder, population, gate and transition head on next-step market targets; it is not a separate self-supervised foundation-model objective.

## Initial joint training

```bash
python -m sams.pretrain --config configs/pretrain.yaml --data data/optiver.parquet --output runs/pretrain --device cuda
```

Batch size is per process. Effective global batch is batch_size × world_size. AMP is opt-in through training.amp and requires CUDA. The supplied default is full precision.

AdamW, gradient clipping and an epoch cosine scheduler are used. This differs from the legacy notebook's scheduler. Model parameter names preserve the extracted architecture, but notebook checkpoint files are not automatically compatible with the versioned packaged checkpoint format.

## Resume at an epoch boundary

```bash
python -m sams.pretrain --config configs/pretrain.yaml --data data/optiver.parquet --output runs/pretrain --device cuda --resume runs/pretrain/latest.pt
```

Resume requires the identical model/training configuration, data SHA-256, process count and max-steps setting. The checkpoint includes optimizer, scheduler, AMP scaler, normalization, epoch, global step and per-rank random state. Mid-epoch continuation is not implemented.

For a controlled interruption, use --stop-after 1 while leaving the configured total epoch count unchanged. Resume without --stop-after to continue the same scheduler horizon. Do not increase epochs in the config and call it an exact resume.

## Encoder-frozen adaptation

```bash
python -m sams.pretrain --config configs/participants.yaml --data data/new-period.parquet --output runs/adaptation --device cuda --initialize-from runs/pretrain/best.pt
```

The model configuration must match the initialization checkpoint. Encoder weights and dropout are frozen; population, gate and transition remain trainable. Optimizer/scheduler state starts fresh and normalization is inherited.

This is a training primitive. It does not implement daily orchestration, bounded online updates, acceptance gates or rollback.

## Evaluation

```bash
python -m sams.evaluate --checkpoint runs/pretrain/best.pt --data data/optiver.parquet --output runs/pretrain/test.json --device cuda
```

Checkpoint selection uses validation NLL; the separate command evaluates test windows. Reported metrics are normalized constant-free Gaussian NLL, per-target MAE/RMSE in original target units, and nonzero-return direction accuracy with a majority-sign baseline.

Evaluation uses deterministic mean actions and observed histories. Overlapping windows repeat some market times; metrics are window-token weighted, not independent-observation confidence estimates. NLL may be negative because the Gaussian constant is omitted. These are predictive diagnostics, not P&L or simulator-fidelity certification.
