"""Teacher-forced held-out metrics; this is not a trading-return evaluator."""

import argparse
import json
from pathlib import Path
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from . import checkpoint
from .config import Config
from .data import Statistics, WindowDataset, load_frame
from .distributed import Context, EvaluationSampler
from .model import MarketSimulationModel
from .objectives import gaussian_nll
from .schema import TARGETS


def supervised(model, x, warmup, deterministic=False):
    output = model(x[:, :-1], deterministic=deterministic, return_sequence=True)
    return {
        key: value[:, warmup:]
        for key, value in output.items()
        if key not in ("hidden", "previous_action", "latent_state")
    }


@torch.no_grad()
def metrics(model, dataset, config, stats, context):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=EvaluationSampler(dataset, context.rank, context.world_size),
    )
    # tokens, nll, six absolute errors, six squared errors, direction correct/count/up
    totals = torch.zeros(17, dtype=torch.float64, device=context.device)
    scale = torch.tensor(stats.y_std, device=context.device)
    center = torch.tensor(stats.y_mean, device=context.device)
    for x, y in loader:
        x, y = x.to(context.device), y.to(context.device)[:, config.training.warmup_len : -1]
        output = supervised(model, x, config.training.warmup_len, deterministic=True)
        prediction = output["state_mean"]
        count = y.shape[0] * y.shape[1]
        error = (prediction - y) * scale
        actual_return = (y * scale + center)[..., 0]
        predicted_return = (prediction * scale + center)[..., 0]
        nonzero = actual_return != 0
        totals[0] += count
        totals[1] += gaussian_nll(y, prediction, output["state_log_std"]) * count
        totals[2:8] += error.abs().sum((0, 1))
        totals[8:14] += error.square().sum((0, 1))
        totals[14] += ((predicted_return.sign() == actual_return.sign()) & nonzero).sum()
        totals[15] += nonzero.sum()
        totals[16] += (actual_return > 0).sum()
    if context.world_size > 1:
        dist.all_reduce(totals)
    values = totals.cpu().tolist()
    count, directions = values[0], values[15]
    if not count or not torch.isfinite(totals).all():
        raise ValueError("Empty or nonfinite evaluation")
    return {
        "tokens": int(count),
        "normalized_nll": values[1] / count,
        "mae": dict(zip(TARGETS, [v / count for v in values[2:8]])),
        "rmse": dict(zip(TARGETS, [(v / count) ** 0.5 for v in values[8:14]])),
        "return_direction_accuracy": values[14] / directions if directions else None,
        "return_majority_baseline": max(values[16], directions - values[16]) / directions
        if directions
        else None,
        "nonzero_return_tokens": int(directions),
        "mode": "teacher_forced_mean_actions; overlapping windows; constant-free NLL",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists")
    torch.set_num_threads(1)
    context = Context.initialize(args.device)
    try:
        state = checkpoint.load(args.checkpoint)
        config = Config.from_dict(state["config"])
        stats = Statistics(**state["statistics"])
        dataset = WindowDataset(load_frame(args.data), "test", config.training.sequence_len, stats)
        model = MarketSimulationModel(config.model).to(context.device)
        model.load_state_dict(state["model"])
        result = metrics(model, dataset, config, stats, context)
        if context.rank == 0:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
    finally:
        context.close()


if __name__ == "__main__":
    main()
