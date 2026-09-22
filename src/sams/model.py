"""Notebook-derived shared encoder, independent GRU population and transition head.

Parameter names and equations match the reference notebook; configuration is explicit.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
from .schema import FEATURES


class Position(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]


class ConfigurableMarketEncoder(nn.Module):
    """
    Shared market encoder.

    The current experiment uses a causal Transformer, but causality,
    layer count, head count, width, and the optional external channel
    are configuration parameters.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if self.config.encoder_type != "transformer":
            raise ValueError(
                "This Notebook currently implements encoder_type='transformer'. Other encoder types require a separate implementation."
            )
        self.market_projection = nn.Linear(len(FEATURES), self.config.d_model)
        self.external_projection = (
            nn.Linear(self.config.external_feature_dim, self.config.d_model)
            if self.config.external_feature_dim > 0
            else None
        )
        self.source_fusion = (
            nn.Linear(2 * self.config.d_model, self.config.d_model)
            if self.config.external_feature_dim > 0
            else None
        )
        self.position = Position(self.config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=self.config.norm_first,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, self.config.layers)
        self.output_projection = nn.Linear(self.config.d_model, self.config.latent_dim)

    def forward(self, market_features, external_features=None):
        hidden = self.market_projection(market_features)
        if self.external_projection is not None:
            if external_features is None:
                external_features = torch.zeros(
                    *market_features.shape[:-1],
                    self.config.external_feature_dim,
                    device=market_features.device,
                    dtype=market_features.dtype,
                )
            external_hidden = self.external_projection(external_features)
            hidden = self.source_fusion(torch.cat([hidden, external_hidden], dim=-1))
        hidden = self.position(hidden)
        attention_mask = None
        if self.config.encoder_causal:
            sequence_len = market_features.size(1)
            attention_mask = torch.triu(
                torch.ones(
                    sequence_len, sequence_len, dtype=torch.bool, device=market_features.device
                ),
                diagonal=1,
            )
        hidden = self.transformer(hidden, mask=attention_mask)
        return self.output_projection(hidden)


class VectorizedAgentPopulation(nn.Module):
    """
    K independent recurrent agents.

    Agents share the encoder output h_t but do not receive other
    agents' hidden states or actions at the same time step.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.ids = [f"latent_agent_{index}" for index in range(self.config.num_agents)]
        input_dim = self.config.latent_dim + self.config.action_dim
        self.weight_ih = nn.Parameter(
            torch.empty(self.config.num_agents, 3 * self.config.agent_hidden, input_dim)
        )
        self.weight_hh = nn.Parameter(
            torch.empty(
                self.config.num_agents, 3 * self.config.agent_hidden, self.config.agent_hidden
            )
        )
        self.bias_ih = nn.Parameter(
            torch.zeros(self.config.num_agents, 3 * self.config.agent_hidden)
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(self.config.num_agents, 3 * self.config.agent_hidden)
        )
        self.body_weight = nn.Parameter(
            torch.empty(self.config.num_agents, self.config.agent_hidden, self.config.agent_hidden)
        )
        self.body_bias = nn.Parameter(torch.zeros(self.config.num_agents, self.config.agent_hidden))
        self.action_mean_weight = nn.Parameter(
            torch.empty(self.config.num_agents, self.config.action_dim, self.config.agent_hidden)
        )
        self.action_mean_bias = nn.Parameter(
            torch.zeros(self.config.num_agents, self.config.action_dim)
        )
        self.action_log_std_weight = nn.Parameter(
            torch.empty(self.config.num_agents, self.config.action_dim, self.config.agent_hidden)
        )
        self.action_log_std_bias = nn.Parameter(
            torch.zeros(self.config.num_agents, self.config.action_dim)
        )
        self.group_gate = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.latent_dim),
            nn.GELU(),
            nn.Linear(self.config.latent_dim, self.config.num_agents),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for index in range(self.config.num_agents):
            nn.init.xavier_uniform_(self.weight_ih[index])
            nn.init.orthogonal_(self.weight_hh[index])
            nn.init.xavier_uniform_(self.body_weight[index])
            nn.init.xavier_uniform_(self.action_mean_weight[index])
            nn.init.xavier_uniform_(self.action_log_std_weight[index])

    def initial_state(self, batch_size, device):
        hidden = torch.zeros(
            batch_size, self.config.num_agents, self.config.agent_hidden, device=device
        )
        previous_action = torch.zeros(
            batch_size, self.config.num_agents, self.config.action_dim, device=device
        )
        return (hidden, previous_action)

    def gru_step(self, agent_input, hidden):
        input_gates = torch.einsum(
            "bkd,khd->bkh", agent_input, self.weight_ih
        ) + self.bias_ih.unsqueeze(0)
        hidden_gates = torch.einsum(
            "bkd,khd->bkh", hidden, self.weight_hh
        ) + self.bias_hh.unsqueeze(0)
        input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_gate = torch.tanh(input_new + reset_gate * hidden_new)
        return new_gate + update_gate * (hidden - new_gate)

    def forward(self, market_state, hidden, previous_action, deterministic=False):
        market_by_agent = market_state.unsqueeze(1).expand(-1, self.config.num_agents, -1)
        agent_input = torch.cat([market_by_agent, previous_action], dim=-1)
        next_hidden = self.gru_step(agent_input, hidden)
        body = F.gelu(
            torch.einsum("bkd,khd->bkh", next_hidden, self.body_weight)
            + self.body_bias.unsqueeze(0)
        )
        means = torch.einsum(
            "bkh,kah->bka", body, self.action_mean_weight
        ) + self.action_mean_bias.unsqueeze(0)
        log_stds = (
            torch.einsum("bkh,kah->bka", body, self.action_log_std_weight)
            + self.action_log_std_bias.unsqueeze(0)
        ).clamp(-5.0, 2.0)
        actions = means if deterministic else means + log_stds.exp() * torch.randn_like(means)
        weights = torch.softmax(self.group_gate(market_state), dim=-1)
        aggregate_action = (actions * weights.unsqueeze(-1)).sum(dim=1)
        return {
            "actions": actions,
            "means": means,
            "log_stds": log_stds,
            "weights": weights,
            "aggregate_action": aggregate_action,
            "hidden": next_hidden,
            "previous": actions,
        }


class MarketTransitionHead(nn.Module):
    """
    Maps the weighted population action to six next-step
    market-transition distributions.

    There is intentionally no direct h_t input.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.body = nn.Sequential(
            nn.Linear(self.config.action_dim, 192), nn.GELU(), nn.Linear(192, 192), nn.GELU()
        )
        self.state_mean = nn.Linear(192, self.config.target_dim)
        self.state_log_std = nn.Linear(192, self.config.target_dim)

    def forward(self, aggregate_action):
        hidden = self.body(aggregate_action)
        return {
            "state_mean": self.state_mean(hidden),
            "state_log_std": self.state_log_std(hidden).clamp(-5.0, 2.0),
        }


class MarketSimulationModel(nn.Module):
    """Shared causal perception, independent recurrent state, action-only transition.

    With return_sequence=True, predict after each input token. The trainer
    passes all but the alignment row and supervises only the configured tail.
    Hidden state must never be carried across unrelated market buckets.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = ConfigurableMarketEncoder(self.config)
        self.population = VectorizedAgentPopulation(self.config)
        self.transition = MarketTransitionHead(self.config)

    def forward(
        self,
        market_sequence,
        external_sequence=None,
        hidden=None,
        previous_action=None,
        deterministic=False,
        return_sequence=False,
    ):
        if market_sequence.ndim != 3 or market_sequence.shape[-1] != len(FEATURES):
            raise ValueError("market_sequence must have shape [batch, time, 19]")
        if not 0 < market_sequence.shape[1] <= 1024:
            raise ValueError("Sequence length must be between 1 and 1024")
        if (hidden is None) != (previous_action is None):
            raise ValueError("Provide both hidden and previous_action or neither")
        latents = self.encoder(market_sequence, external_sequence)
        if hidden is None:
            hidden, previous_action = self.population.initial_state(
                market_sequence.shape[0], market_sequence.device
            )
        predictions = []
        for index in range(market_sequence.shape[1]):
            population = self.population(latents[:, index], hidden, previous_action, deterministic)
            hidden, previous_action = population["hidden"], population["previous"]
            if return_sequence or index == market_sequence.shape[1] - 1:
                predictions.append(
                    {
                        **self.transition(population["aggregate_action"]),
                        "action_means": population["means"],
                        "actions": population["actions"],
                        "weights": population["weights"],
                        "aggregate_action": population["aggregate_action"],
                    }
                )
        output = (
            {key: torch.stack([p[key] for p in predictions], dim=1) for key in predictions[0]}
            if return_sequence
            else predictions[-1]
        )
        return {
            **output,
            "hidden": hidden,
            "previous_action": previous_action,
            "latent_state": latents[:, -1],
        }

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()
