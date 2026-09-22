"""Float32 likelihood and population regularizers; no implicit global state."""

import torch
from torch.nn import functional as F


def gaussian_nll(target, mean, log_std):
    """Constant-free diagonal Gaussian NLL, averaged over tokens and targets."""
    log_std = log_std.float().clamp(-5, 2)
    return (0.5 * (target.float() - mean.float()).square() * (-2 * log_std).exp() + log_std).mean()


def diversity_loss(means):
    unit = F.normalize(means.float(), dim=-1)
    similarity = unit @ unit.transpose(-1, -2)
    count = means.shape[-2]
    mask = 1 - torch.eye(count, device=means.device)
    return (similarity * mask).square().sum() / max(1, means.shape[0] * count * (count - 1))


def objective(output, targets, config):
    nll = gaussian_nll(targets, output["state_mean"], output["state_log_std"])
    actions = output["action_means"]
    diversity = diversity_loss(actions.flatten(0, 1))
    weights = output["weights"].float()
    balance = (weights - 1 / weights.shape[-1]).square().mean()
    total = nll + config.lambda_diversity * diversity + config.lambda_balance * balance
    return {"loss": total, "nll": nll, "diversity": diversity, "balance": balance}
