import torch
from sams.model import MarketSimulationModel
from sams.objectives import objective, diversity_loss
from sams.evaluate import supervised


def test_causal_encoder(config):
    model = MarketSimulationModel(config.model).eval()
    x = torch.randn(2, 8, 19)
    changed = x.clone()
    changed[:, 5:] += 100
    torch.testing.assert_close(model.encoder(x)[:, :5], model.encoder(changed)[:, :5])


def test_shapes_and_all_parameters_train(config):
    model = MarketSimulationModel(config.model)
    x = torch.randn(2, config.training.sequence_len, 19)
    result = supervised(model, x, config.training.warmup_len)
    assert result["state_mean"].shape == (2, 4, 6)
    torch.testing.assert_close(result["weights"].sum(-1), torch.ones(2, 4))
    objective(result, torch.randn(2, 4, 6), config.training)["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_agent_state_is_independent(config):
    population = MarketSimulationModel(config.model).population
    latent = torch.randn(2, config.model.latent_dim)
    hidden, previous = population.initial_state(2, torch.device("cpu"))
    changed = hidden.clone()
    changed[:, 0] += 3
    left = population(latent, hidden, previous, True)
    right = population(latent, changed, previous, True)
    torch.testing.assert_close(left["means"][:, 1:], right["means"][:, 1:])
    assert not torch.equal(left["means"][:, 0], right["means"][:, 0])


def test_encoder_freezing(config):
    model = MarketSimulationModel(config.model)
    model.freeze_encoder()
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.population.parameters())


def test_single_agent_diversity():
    assert diversity_loss(torch.randn(3, 1, 4)).item() == 0
