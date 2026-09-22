import pytest
import torch
from sams import checkpoint
from sams.data import Statistics
from sams.model import MarketSimulationModel


def test_roundtrip_and_schema(tmp_path, config, frame):
    model = MarketSimulationModel(config.model).eval()
    path = tmp_path / "model.pt"
    checkpoint.save(
        path,
        {
            "config": config.to_dict(),
            "statistics": Statistics.fit(frame).to_dict(),
            "model": model.state_dict(),
        },
    )
    state = checkpoint.load(path)
    restored = MarketSimulationModel(config.model).eval()
    restored.load_state_dict(state["model"])
    x = torch.randn(2, 4, 19)
    torch.testing.assert_close(
        model(x, deterministic=True)["state_mean"], restored(x, deterministic=True)["state_mean"]
    )
    state["features"] = ["incorrect"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="schema"):
        checkpoint.load(path)
