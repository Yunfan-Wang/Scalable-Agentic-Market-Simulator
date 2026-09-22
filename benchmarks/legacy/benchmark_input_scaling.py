
from __future__ import annotations
import argparse, csv, json, math, statistics, time
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BenchConfig:
    market_feature_dim: int = 19
    external_feature_dim: int = 0
    d_model: int = 128
    latent_dim: int = 128
    nhead: int = 8
    layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.0
    norm_first: bool = True
    encoder_causal: bool = True
    num_agents: int = 24
    agent_hidden: int = 64
    action_dim: int = 4
    target_dim: int = 6
    sequence_len: int = 129

class Position(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(dtype=x.dtype, device=x.device)

class ConfigurableMarketEncoder(nn.Module):
    """Faithful dummy of the uploaded shared Transformer interpreter."""
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.cfg = cfg
        self.market_projection = nn.Linear(cfg.market_feature_dim, cfg.d_model)
        self.external_projection = nn.Linear(cfg.external_feature_dim, cfg.d_model) if cfg.external_feature_dim > 0 else None
        self.source_fusion = nn.Linear(2 * cfg.d_model, cfg.d_model) if cfg.external_feature_dim > 0 else None
        self.position = Position(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=cfg.norm_first, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.layers)
        self.output_projection = nn.Linear(cfg.d_model, cfg.latent_dim)
    def forward(self, market_features, external_features=None):
        hidden = self.market_projection(market_features)
        if self.external_projection is not None:
            if external_features is None:
                external_features = torch.zeros(
                    *market_features.shape[:-1], self.cfg.external_feature_dim,
                    device=market_features.device, dtype=market_features.dtype
                )
            ext = self.external_projection(external_features)
            hidden = self.source_fusion(torch.cat([hidden, ext], dim=-1))
        hidden = self.position(hidden)
        mask = None
        if self.cfg.encoder_causal:
            L = market_features.size(1)
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=market_features.device), diagonal=1)
        hidden = self.transformer(hidden, mask=mask)
        return self.output_projection(hidden)

class VectorizedAgentPopulation(nn.Module):
    """K independent vectorized recurrent decision cores, matching the uploaded design."""
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.cfg = cfg
        K, H, A, D = cfg.num_agents, cfg.agent_hidden, cfg.action_dim, cfg.latent_dim + cfg.action_dim
        self.weight_ih = nn.Parameter(torch.empty(K, 3*H, D))
        self.weight_hh = nn.Parameter(torch.empty(K, 3*H, H))
        self.bias_ih = nn.Parameter(torch.zeros(K, 3*H))
        self.bias_hh = nn.Parameter(torch.zeros(K, 3*H))
        self.body_weight = nn.Parameter(torch.empty(K, H, H))
        self.body_bias = nn.Parameter(torch.zeros(K, H))
        self.action_mean_weight = nn.Parameter(torch.empty(K, A, H))
        self.action_mean_bias = nn.Parameter(torch.zeros(K, A))
        self.action_log_std_weight = nn.Parameter(torch.empty(K, A, H))
        self.action_log_std_bias = nn.Parameter(torch.zeros(K, A))
        self.group_gate = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.GELU(), nn.Linear(cfg.latent_dim, K))
        self.reset_parameters()
    def reset_parameters(self):
        for k in range(self.cfg.num_agents):
            nn.init.xavier_uniform_(self.weight_ih[k]); nn.init.orthogonal_(self.weight_hh[k])
            nn.init.xavier_uniform_(self.body_weight[k])
            nn.init.xavier_uniform_(self.action_mean_weight[k]); nn.init.xavier_uniform_(self.action_log_std_weight[k])
    def initial_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.cfg.num_agents, self.cfg.agent_hidden, device=device),
            torch.zeros(batch_size, self.cfg.num_agents, self.cfg.action_dim, device=device),
        )
    def gru_step(self, agent_input, hidden):
        ig = torch.einsum("bkd,khd->bkh", agent_input, self.weight_ih) + self.bias_ih.unsqueeze(0)
        hg = torch.einsum("bkd,khd->bkh", hidden, self.weight_hh) + self.bias_hh.unsqueeze(0)
        ir, iu, inn = ig.chunk(3, dim=-1); hr, hu, hn = hg.chunk(3, dim=-1)
        r = torch.sigmoid(ir + hr); u = torch.sigmoid(iu + hu); n = torch.tanh(inn + r * hn)
        return n + u * (hidden - n)
    def forward(self, market_state, hidden, previous_action, deterministic=True):
        market_by_agent = market_state.unsqueeze(1).expand(-1, self.cfg.num_agents, -1)
        agent_input = torch.cat([market_by_agent, previous_action], dim=-1)
        next_hidden = self.gru_step(agent_input, hidden)
        body = F.gelu(torch.einsum("bkd,khd->bkh", next_hidden, self.body_weight) + self.body_bias.unsqueeze(0))
        means = torch.einsum("bkh,kah->bka", body, self.action_mean_weight) + self.action_mean_bias.unsqueeze(0)
        log_stds = (torch.einsum("bkh,kah->bka", body, self.action_log_std_weight) + self.action_log_std_bias.unsqueeze(0)).clamp(-5, 2)
        actions = means if deterministic else means + log_stds.exp() * torch.randn_like(means)
        weights = torch.softmax(self.group_gate(market_state), dim=-1)
        aggregate_action = (actions * weights.unsqueeze(-1)).sum(dim=1)
        return {"hidden": next_hidden, "previous": actions, "aggregate_action": aggregate_action}

class MarketTransitionHead(nn.Module):
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(cfg.action_dim, 192), nn.GELU(), nn.Linear(192, 192), nn.GELU())
        self.state_mean = nn.Linear(192, cfg.target_dim)
        self.state_log_std = nn.Linear(192, cfg.target_dim)
    def forward(self, aggregate_action):
        h = self.body(aggregate_action)
        return self.state_mean(h), self.state_log_std(h).clamp(-5, 2)

class MarketSimulationModel(nn.Module):
    def __init__(self, cfg: BenchConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConfigurableMarketEncoder(cfg)
        self.population = VectorizedAgentPopulation(cfg)
        self.transition = MarketTransitionHead(cfg)
    def forward(self, market_sequence, external_sequence=None):
        token_latents = self.encoder(market_sequence, external_sequence)
        hidden, previous = self.population.initial_state(market_sequence.size(0), market_sequence.device)
        pop = None
        for t in range(market_sequence.size(1)):
            pop = self.population(token_latents[:, t], hidden, previous, deterministic=True)
            hidden, previous = pop["hidden"], pop["previous"]
        mean, log_std = self.transition(pop["aggregate_action"])
        return mean, log_std

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def benchmark_model(cfg, device, warmup=6, repeats=25):
    torch.manual_seed(7)
    model = MarketSimulationModel(cfg).eval().to(device)
    market = torch.randn(1, cfg.sequence_len, cfg.market_feature_dim, device=device)
    external = torch.randn(1, cfg.sequence_len, cfg.external_feature_dim, device=device) if cfg.external_feature_dim > 0 else None
    with torch.inference_mode():
        for _ in range(warmup):
            model(market, external)
        sync(device)
        samples=[]
        for _ in range(repeats):
            sync(device); t0=time.perf_counter()
            model(market, external)
            sync(device); samples.append((time.perf_counter()-t0)*1000)
    params=sum(p.numel() for p in model.parameters())
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": sorted(samples)[max(0, math.ceil(0.95*len(samples))-1)],
        "parameters": params,
    }

def save_results(rows, csv_path, json_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    Path(json_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")

def save_latency_graph(rows, output_csv):
    """Plot every tested input width explicitly and annotate measured values."""
    labels = [str(row["market_feature_dim"]) for row in rows]
    x_pos = list(range(len(rows)))
    median = [row["median_ms"] for row in rows]
    p95 = [row["p95_ms"] for row in rows]

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.plot(x_pos, median, marker="o", linewidth=2.2, label="Median latency")
    ax.plot(x_pos, p95, marker="o", linewidth=1.5, linestyle="--", label="P95 latency")

    # Categorical positions guarantee that EVERY tested configuration is visible.
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_xlabel("Transformer market input features (exact tested width)")
    ax.set_ylabel("Forward latency (ms)")
    ax.set_title("Latency sensitivity to information-input width")

    for x, y in zip(x_pos, median):
        ax.annotate(f"{y:.2f} ms", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8)
    for x, y in zip(x_pos, p95):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=7)

    cfg = BenchConfig()
    info = (
        f"Fixed: d_model={cfg.d_model} | latent={cfg.latent_dim} | "
        f"layers={cfg.layers} | heads={cfg.nhead} | ff={cfg.ff_dim} | "
        f"K={cfg.num_agents} | agent_hidden={cfg.agent_hidden} | "
        f"seq={cfg.sequence_len} | device={rows[0]['device']}\n"
        "Labels above solid line = median ms; labels below dashed line = P95 ms"
    )
    fig.text(0.5, 0.01, info, ha="center", va="bottom", fontsize=8)

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    png_path = Path(output_csv).with_suffix(".png")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved to: {png_path}")

def main():
    ap=argparse.ArgumentParser(description="Transformer input-width latency benchmark.")
    ap.add_argument("--input-dims", default="19,32,64,128,256,512,1024,2048,4096,8192")
    ap.add_argument("--warmup", type=int, default=6); ap.add_argument("--repeats", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="input_scaling_results.csv")
    args=ap.parse_args(); device=torch.device(args.device)
    rows=[]
    for D in [int(x) for x in args.input_dims.split(",")]:
        cfg=BenchConfig(market_feature_dim=D, num_agents=24)
        r=benchmark_model(cfg, device, args.warmup, args.repeats)
        rows.append({"market_feature_dim":D, **r, "device":str(device)})
        print(rows[-1])
    save_results(rows, args.out, str(Path(args.out).with_suffix(".json")))
    save_latency_graph(rows, args.out)
if __name__=="__main__": main()
