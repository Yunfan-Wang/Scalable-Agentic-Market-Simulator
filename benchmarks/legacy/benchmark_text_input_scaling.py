
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BenchConfig:
    # Structured market branch
    market_feature_dim: int = 19
    market_sequence_len: int = 129

    # Text branch
    vocab_size: int = 32768
    text_sequence_len: int = 1024
    text_d_model: int = 128
    text_nhead: int = 8
    text_layers: int = 4
    text_ff_dim: int = 256

    # Shared market interpreter / latent state
    market_d_model: int = 128
    market_nhead: int = 8
    market_layers: int = 4
    market_ff_dim: int = 256
    latent_dim: int = 128

    # Participant population
    num_agents: int = 24
    agent_hidden: int = 64
    action_dim: int = 4

    # Transition
    target_dim: int = 6


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(
            dtype=x.dtype,
            device=x.device,
        )


class StructuredMarketEncoder(nn.Module):
    """
    Faithful structured-data branch:
    19-dimensional market observations -> causal Transformer state.
    """
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.market_feature_dim, cfg.market_d_model)
        self.pos = SinusoidalPosition(cfg.market_d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.market_d_model,
            nhead=cfg.market_nhead,
            dim_feedforward=cfg.market_ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.market_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.proj(x))
        L = x.size(1)
        causal_mask = torch.triu(
            torch.ones(
                L,
                L,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        return self.encoder(
            h,
            mask=causal_mask,
        )


class TextInformationEncoder(nn.Module):
    """
    Dummy raw-text information branch.

    Random token IDs are used because this benchmark measures computational
    latency rather than semantic quality. The computational structure is what
    matters: token embedding -> positional encoding -> Transformer -> pooled
    context representation.
    """
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            cfg.vocab_size,
            cfg.text_d_model,
        )
        self.pos = SinusoidalPosition(
            cfg.text_d_model,
            max_len=max(8192, cfg.text_sequence_len),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.text_d_model,
            nhead=cfg.text_nhead,
            dim_feedforward=cfg.text_ff_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=cfg.text_layers,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.pos(self.embedding(token_ids))

        # Text information is interpreted as a context block available
        # at the simulation anchor, so bidirectional self-attention is used.
        h = self.encoder(h)

        # Mean pooling keeps the downstream fusion dimension independent
        # of the number of text tokens.
        return h.mean(dim=1)


class SharedInformationFusion(nn.Module):
    """
    Fuse the latest structured market state with one interpreted text context,
    then produce the shared latent market state h_t.
    """
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        fusion_dim = cfg.market_d_model + cfg.text_d_model

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, cfg.market_d_model),
            nn.GELU(),
            nn.Linear(cfg.market_d_model, cfg.latent_dim),
        )

    def forward(
        self,
        market_state: torch.Tensor,
        text_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(
            torch.cat(
                [market_state, text_state],
                dim=-1,
            )
        )


class VectorizedAgentPopulation(nn.Module):
    """
    K independent vectorized recurrent participant decision cores.

    Each participant reads the same shared latent state but retains its own
    recurrent memory and previous action.
    """
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.cfg = cfg

        K = cfg.num_agents
        H = cfg.agent_hidden
        A = cfg.action_dim
        D = cfg.latent_dim + cfg.action_dim

        self.weight_ih = nn.Parameter(
            torch.empty(K, 3 * H, D)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(K, 3 * H, H)
        )
        self.bias_ih = nn.Parameter(
            torch.zeros(K, 3 * H)
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(K, 3 * H)
        )

        self.body_weight = nn.Parameter(
            torch.empty(K, H, H)
        )
        self.body_bias = nn.Parameter(
            torch.zeros(K, H)
        )

        self.action_weight = nn.Parameter(
            torch.empty(K, A, H)
        )
        self.action_bias = nn.Parameter(
            torch.zeros(K, A)
        )

        self.gate = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, K),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.cfg.num_agents):
            nn.init.xavier_uniform_(self.weight_ih[k])
            nn.init.orthogonal_(self.weight_hh[k])
            nn.init.xavier_uniform_(self.body_weight[k])
            nn.init.xavier_uniform_(self.action_weight[k])

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ):
        hidden = torch.zeros(
            batch_size,
            self.cfg.num_agents,
            self.cfg.agent_hidden,
            device=device,
        )
        previous_action = torch.zeros(
            batch_size,
            self.cfg.num_agents,
            self.cfg.action_dim,
            device=device,
        )
        return hidden, previous_action

    def gru_step(
        self,
        agent_input: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        input_gates = (
            torch.einsum(
                "bkd,khd->bkh",
                agent_input,
                self.weight_ih,
            )
            + self.bias_ih.unsqueeze(0)
        )
        hidden_gates = (
            torch.einsum(
                "bkd,khd->bkh",
                hidden,
                self.weight_hh,
            )
            + self.bias_hh.unsqueeze(0)
        )

        ir, iz, inn = input_gates.chunk(
            3,
            dim=-1,
        )
        hr, hz, hn = hidden_gates.chunk(
            3,
            dim=-1,
        )

        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        new = torch.tanh(
            inn + reset * hn
        )

        return new + update * (
            hidden - new
        )

    def forward(
        self,
        latent_state: torch.Tensor,
        hidden: torch.Tensor,
        previous_action: torch.Tensor,
    ):
        shared = latent_state.unsqueeze(1).expand(
            -1,
            self.cfg.num_agents,
            -1,
        )

        agent_input = torch.cat(
            [shared, previous_action],
            dim=-1,
        )

        next_hidden = self.gru_step(
            agent_input,
            hidden,
        )

        body = F.gelu(
            torch.einsum(
                "bkd,khd->bkh",
                next_hidden,
                self.body_weight,
            )
            + self.body_bias.unsqueeze(0)
        )

        actions = (
            torch.einsum(
                "bkh,kah->bka",
                body,
                self.action_weight,
            )
            + self.action_bias.unsqueeze(0)
        )

        gate_weights = torch.softmax(
            self.gate(latent_state),
            dim=-1,
        )

        aggregate_action = (
            actions
            * gate_weights.unsqueeze(-1)
        ).sum(dim=1)

        return {
            "hidden": next_hidden,
            "previous_action": actions,
            "aggregate_action": aggregate_action,
        }


class MarketTransitionHead(nn.Module):
    """
    Action-only transition bottleneck:
    aggregate participant action -> six market-state target distributions.
    """
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(cfg.action_dim, 192),
            nn.GELU(),
            nn.Linear(192, 192),
            nn.GELU(),
        )
        self.mean = nn.Linear(
            192,
            cfg.target_dim,
        )
        self.log_std = nn.Linear(
            192,
            cfg.target_dim,
        )

    def forward(
        self,
        aggregate_action: torch.Tensor,
    ):
        h = self.body(
            aggregate_action
        )
        return (
            self.mean(h),
            self.log_std(h).clamp(-5, 2),
        )


class TextAugmentedSAMS(nn.Module):
    """
    Dummy full SAMS architecture for computational benchmarking:

    structured market data  ─┐
                             ├─> shared latent interpretation
    raw text information ────┘
                                      |
                                      v
                          K recurrent participant agents
                                      |
                                      v
                              weighted aggregate action
                                      |
                                      v
                          probabilistic market transition
    """
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.cfg = cfg

        self.market_encoder = StructuredMarketEncoder(
            cfg
        )
        self.text_encoder = TextInformationEncoder(
            cfg
        )
        self.fusion = SharedInformationFusion(
            cfg
        )
        self.population = VectorizedAgentPopulation(
            cfg
        )
        self.transition = MarketTransitionHead(
            cfg
        )

    def forward(
        self,
        market_sequence: torch.Tensor,
        token_ids: torch.Tensor,
    ):
        market_tokens = self.market_encoder(
            market_sequence
        )

        text_context = self.text_encoder(
            token_ids
        )

        # Use the latest structured market representation
        # at the simulation anchor.
        latest_market_state = market_tokens[:, -1]

        latent_state = self.fusion(
            latest_market_state,
            text_context,
        )

        hidden, previous_action = (
            self.population.initial_state(
                market_sequence.size(0),
                market_sequence.device,
            )
        )

        population = self.population(
            latent_state,
            hidden,
            previous_action,
        )

        mean, log_std = self.transition(
            population["aggregate_action"]
        )

        return mean, log_std


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    cfg: BenchConfig,
    device: torch.device,
    warmup: int,
    repeats: int,
):
    torch.manual_seed(7)

    model = TextAugmentedSAMS(
        cfg
    ).eval().to(device)

    market = torch.randn(
        1,
        cfg.market_sequence_len,
        cfg.market_feature_dim,
        device=device,
    )

    token_ids = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(
            1,
            cfg.text_sequence_len,
        ),
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

    with torch.inference_mode():
        for _ in range(warmup):
            model(
                market,
                token_ids,
            )

        synchronize(device)

        samples = []

        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()

            model(
                market,
                token_ids,
            )

            synchronize(device)

            samples.append(
                (
                    time.perf_counter()
                    - start
                )
                * 1000.0
            )

    peak_memory_mb = None

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / 1024**2
        )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    ordered = sorted(samples)

    p95_index = max(
        0,
        math.ceil(
            0.95 * len(ordered)
        )
        - 1,
    )

    return {
        "median_ms": statistics.median(
            samples
        ),
        "mean_ms": statistics.mean(
            samples
        ),
        "p95_ms": ordered[p95_index],
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameters": parameters,
    }


def save_results(
    rows,
    csv_path: str,
):
    csv_path = Path(csv_path)
    json_path = csv_path.with_suffix(
        ".json"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_latency_graph(rows, csv_path: str):
    """Plot every tested token length and annotate exact latency values."""
    labels = [str(row["text_tokens"]) for row in rows]
    x_pos = list(range(len(rows)))
    median = [row["median_ms"] for row in rows]
    p95 = [row["p95_ms"] for row in rows]

    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.plot(x_pos, median, marker="o", linewidth=2.3, label="Median latency")
    ax.plot(x_pos, p95, marker="o", linewidth=1.6, linestyle="--", label="P95 latency")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Raw-text context length (exact tested tokens)")
    ax.set_ylabel("Full-model forward latency (ms)")
    ax.set_title("SAMS latency sensitivity to raw-text market information")

    for x, y in zip(x_pos, median):
        ax.annotate(f"{y:.2f} ms", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)
    for x, y in zip(x_pos, p95):
        ax.annotate(f"P95 {y:.2f}", (x, y), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=7)

    first = rows[0]
    info = (
        f"Fixed: structured_input=19 | market_seq=129 | text_d_model={first['text_d_model']} | "
        f"text_layers={first['text_layers']} | latent={first['latent_dim']} | "
        f"K={first['num_agents']} | device={first['device']}"
    )
    fig.text(0.5, 0.012, info, ha="center", va="bottom", fontsize=8)

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    png_path = Path(csv_path).with_suffix(".png")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Latency graph saved to: {png_path}")

def save_memory_graph(rows, csv_path: str):
    """Plot exact GPU memory for every tested token length."""
    if any(row["peak_gpu_memory_mb"] is None for row in rows):
        return

    labels = [str(row["text_tokens"]) for row in rows]
    x_pos = list(range(len(rows)))
    memory = [row["peak_gpu_memory_mb"] for row in rows]

    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.plot(x_pos, memory, marker="o", linewidth=2.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Raw-text context length (exact tested tokens)")
    ax.set_ylabel("Peak allocated GPU memory (MB)")
    ax.set_title("GPU-memory sensitivity to raw-text context length")

    for x, y in zip(x_pos, memory):
        ax.annotate(f"{y:.1f} MB", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)

    first = rows[0]
    info = (
        f"Fixed: text_d_model={first['text_d_model']} | text_layers={first['text_layers']} | "
        f"latent={first['latent_dim']} | K={first['num_agents']} | device={first['device']}"
    )
    fig.text(0.5, 0.012, info, ha="center", va="bottom", fontsize=8)

    ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    csv_path = Path(csv_path)
    memory_png = csv_path.parent / (csv_path.stem + "_memory.png")
    fig.savefig(memory_png, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Memory graph saved to: {memory_png}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SAMS inference latency as raw-text "
            "market-information context length increases."
        )
    )

    parser.add_argument(
        "--text-tokens",
        default=(
            "128,256,512,1024,2048,4096,8192"
        ),
        help=(
            "Comma-separated raw-text context lengths."
        ),
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=24,
        help=(
            "Number of recurrent participant agents."
        ),
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--text-d-model",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--text-layers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "text_input_scaling_results.csv"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build "
            "does not have CUDA available. Install a CUDA-enabled "
            "PyTorch wheel or run with --device cpu."
        )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    token_lengths = [
        int(value)
        for value in args.text_tokens.split(",")
    ]

    rows = []

    for token_count in token_lengths:
        cfg = BenchConfig(
            text_sequence_len=token_count,
            num_agents=args.agents,
            latent_dim=args.latent_dim,
            text_d_model=args.text_d_model,
            text_layers=args.text_layers,
        )

        result = benchmark(
            cfg,
            device,
            args.warmup,
            args.repeats,
        )

        row = {
            "text_tokens": token_count,
            "num_agents": cfg.num_agents,
            "text_d_model": cfg.text_d_model,
            "text_layers": cfg.text_layers,
            "latent_dim": cfg.latent_dim,
            "median_ms": result["median_ms"],
            "mean_ms": result["mean_ms"],
            "p95_ms": result["p95_ms"],
            "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
            "parameters": result["parameters"],
            "device": str(device),
        }

        rows.append(
            row
        )

        print(
            row
        )

    save_results(
        rows,
        args.out,
    )

    save_latency_graph(
        rows,
        args.out,
    )

    save_memory_graph(
        rows,
        args.out,
    )


if __name__ == "__main__":
    main()
