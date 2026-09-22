"""Small, strict configuration objects shared by training and inference."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml


@dataclass(frozen=True)
class ModelConfig:
    encoder_type: str = "transformer"
    encoder_causal: bool = True
    external_feature_dim: int = 0
    d_model: int = 128
    latent_dim: int = 128
    nhead: int = 8
    layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    norm_first: bool = True
    num_agents: int = 24
    agent_hidden: int = 64
    action_dim: int = 4
    target_dim: int = 6

    def __post_init__(self):
        for name in (
            "d_model",
            "latent_dim",
            "nhead",
            "layers",
            "ff_dim",
            "num_agents",
            "agent_hidden",
            "action_dim",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.nhead or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by nhead")
        if not 0 <= self.dropout < 1 or self.target_dim != 6:
            raise ValueError("dropout must be in [0,1); the data contract has six targets")
        if self.encoder_type != "transformer" or not self.encoder_causal:
            raise ValueError("The packaged training path requires a causal transformer")
        if self.external_feature_dim != 0:
            raise ValueError(
                "External feature ingestion is not implemented in the packaged pipeline"
            )


@dataclass(frozen=True)
class TrainingConfig:
    warmup_len: int = 96
    prediction_len: int = 32
    batch_size: int = 16
    epochs: int = 30
    lr: float = 0.0006
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    lambda_diversity: float = 0.03
    lambda_balance: float = 0.005
    num_workers: int = 0
    seed: int = 42
    amp: bool = False
    stage: str = "joint"

    @property
    def sequence_len(self):
        return self.warmup_len + self.prediction_len + 1

    def __post_init__(self):
        for name in ("batch_size", "epochs", "prediction_len"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_len", "num_workers", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.sequence_len > 599:
            raise ValueError("A sequence must fit within a 599-row bucket")
        if self.stage not in ("joint", "participants"):
            raise ValueError("stage must be joint or participants")
        for name in ("lr", "weight_decay", "grad_clip", "lambda_diversity", "lambda_balance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.lr or not self.grad_clip:
            raise ValueError("lr and grad_clip must be positive")


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) - {"model", "training"}:
            raise ValueError("Expected model and training configuration sections")
        return cls(
            ModelConfig(**values.get("model", {})), TrainingConfig(**values.get("training", {}))
        )

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
