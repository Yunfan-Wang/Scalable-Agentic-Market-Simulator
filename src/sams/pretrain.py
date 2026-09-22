"""Joint initial training or encoder-frozen participant adaptation under torchrun."""

import argparse
import json
import platform
import random
import subprocess
from pathlib import Path
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from . import checkpoint
from .config import Config
from .data import Statistics, WindowDataset, load_frame, file_sha256
from .distributed import Context
from .evaluate import metrics, supervised
from .model import MarketSimulationModel
from .objectives import objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from")
    parser.add_argument(
        "--max-steps", type=int, help="Limit training steps per epoch for smoke runs"
    )
    parser.add_argument(
        "--stop-after", type=int, help="Stop after this completed epoch; retain scheduler horizon"
    )
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        parser.error("Choose resume or initialize-from")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    config = Config.load(args.config)
    cfg = config.training
    if args.stop_after is not None and not 1 <= args.stop_after <= cfg.epochs:
        parser.error("stop-after must be between 1 and configured epochs")
    if cfg.stage == "participants" and not (args.initialize_from or args.resume):
        parser.error("participants stage requires initialize-from or resume")
    torch.set_num_threads(1)
    context = Context.initialize(args.device)
    try:
        if cfg.amp and context.device.type != "cuda":
            raise ValueError("AMP requires CUDA; disable it for CPU")
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        root = Path(args.output)
        if root.exists() and not args.resume:
            raise ValueError("Output exists; choose a new directory or resume")
        frame = load_frame(args.data)
        fingerprint = file_sha256(args.data)
        state = (
            checkpoint.load(args.resume or args.initialize_from)
            if (args.resume or args.initialize_from)
            else None
        )
        stats = Statistics(**state["statistics"]) if state else Statistics.fit(frame)
        model = MarketSimulationModel(config.model).to(context.device)
        if state:
            if state["config"]["model"] != config.to_dict()["model"]:
                raise ValueError("Initialization model configuration mismatch")
            model.load_state_dict(state["model"])
        if cfg.stage == "participants":
            model.freeze_encoder()
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
        train = WindowDataset(frame, "train", cfg.sequence_len, stats)
        validation = WindowDataset(frame, "validation", cfg.sequence_len, stats)
        sampler = DistributedSampler(
            train,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=True,
        )
        loader = DataLoader(
            train,
            batch_size=cfg.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=cfg.num_workers,
        )
        if not len(loader):
            raise ValueError("Training data is too small for batch_size * world_size")
        wrapped = (
            DistributedDataParallel(
                model, device_ids=[context.device.index] if context.device.type == "cuda" else None
            )
            if context.world_size > 1
            else model
        )
        epoch_start, step, best = 0, 0, float("inf")
        torch.manual_seed(cfg.seed + context.rank)
        if args.resume:
            if (
                state["config"] != config.to_dict()
                or state["data_sha256"] != fingerprint
                or state["world_size"] != context.world_size
                or state["max_steps"] != args.max_steps
            ):
                raise ValueError("Resume requires identical config, data, world size and max-steps")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            epoch_start, step, best = state["epoch"], state["step"], state["best"]
            rng = state["rng"][context.rank]
            torch.set_rng_state(rng["torch"])
            random.setstate(rng["python"])
            if context.device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"], context.device)
        if context.rank == 0:
            root.mkdir(parents=True, exist_ok=True)

            def git_value(*parts):
                result = subprocess.run(
                    ["git", *parts], capture_output=True, text=True, check=False
                )
                return result.stdout.strip() if result.returncode == 0 else "unavailable"

            manifest = {
                "config": config.to_dict(),
                "data_sha256": fingerprint,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "device": str(context.device),
                "world_size": context.world_size,
                "git_commit": git_value("rev-parse", "HEAD"),
                "git_dirty": bool(git_value("status", "--porcelain")),
                "max_steps_per_epoch": args.max_steps,
                "resume": args.resume,
                "initialize_from": args.initialize_from,
            }
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for epoch in range(epoch_start, args.stop_after or cfg.epochs):
            sampler.set_epoch(epoch)
            model.train()
            if cfg.stage == "participants":
                model.encoder.eval()
            for batch_index, (x, y) in enumerate(loader):
                if args.max_steps is not None and batch_index >= args.max_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                x, y = x.to(context.device), y.to(context.device)[:, cfg.warmup_len : -1]
                with torch.autocast(context.device.type, enabled=cfg.amp):
                    losses = objective(supervised(wrapped, x, cfg.warmup_len), y, cfg)
                if not context.all_finite(torch.isfinite(losses["loss"]).all()):
                    raise FloatingPointError("Nonfinite loss on at least one rank")
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not context.all_finite(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite gradient on at least one rank")
                scaler.step(optimizer)
                scaler.update()
                step += 1
            scheduler.step()
            result = metrics(model, validation, config, stats, context)
            improved = result["normalized_nll"] < best
            best = min(best, result["normalized_nll"])
            rng = {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state(context.device)
                if context.device.type == "cuda"
                else None,
            }
            rngs = [None] * context.world_size
            if context.world_size > 1:
                dist.all_gather_object(rngs, rng)
            else:
                rngs = [rng]
            if context.rank == 0:
                payload = {
                    "config": config.to_dict(),
                    "statistics": stats.to_dict(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "step": step,
                    "best": best,
                    "rng": rngs,
                    "world_size": context.world_size,
                    "data_sha256": fingerprint,
                    "max_steps": args.max_steps,
                }
                checkpoint.save(root / "latest.pt", payload)
                if improved:
                    checkpoint.save(root / "best.pt", payload)
                with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"epoch": epoch + 1, "step": step, **result}) + "\n")
                print(
                    f"epoch={epoch + 1} step={step} validation_nll={result['normalized_nll']:.6f}",
                    flush=True,
                )
    finally:
        context.close()


if __name__ == "__main__":
    main()
