# Architecture

## A shared interpretation, independent responses

At time t, a causal Transformer maps the observed feature history to latent state h(t). Every agent sees h(t), but maintains a private GRU state and previous action. Agent weights are independent tensors, vectorized along the population dimension.

Each agent emits Gaussian action parameters. Training uses reparameterized samples; deterministic evaluation uses action means. A softmax gate conditioned on the shared latent weights the actions. A two-hidden-layer MLP receives only this aggregate action and predicts six means and log standard deviations.

The transition bottleneck prevents a direct encoder-to-transition shortcut. It does not establish that learned agents are causal market participants.

| Component | Reference configuration |
|---|---|
| Input | 19 features, one-second grid |
| Encoder | 4 Transformer layers, width 128, 8 attention heads, FFN 256 |
| Population | 24 separately parameterized GRUs, hidden width 64 |
| Action | 4 latent dimensions per agent; log std clamped to [-5, 2] |
| Gate | Softmax weights summing to one |
| Transition | Aggregate action → 192 → 192 → six means and log standard deviations |
| Supervision | 32 steps after 96 warmup tokens |

## Training objective

Loss = constant-free diagonal Gaussian NLL + 0.03 × diversity penalty + 0.005 × gate-balance penalty.

The diversity penalty is mean squared off-diagonal cosine similarity between agent action means. The balance term penalizes deviation from uniform gate weights. Neither term guarantees economically meaningful specialization.

Likelihood arithmetic uses float32. Warmup steps retain gradients. Each window starts with fresh recurrent state; state must not cross unrelated buckets.

## Read the code

- [Model](../src/sams/model.py)
- [Objectives](../src/sams/objectives.py)
- [Strict configuration](../src/sams/config.py)
- [Causality and independence tests](../tests/test_model.py)

The public forward call accepts [batch, time, 19] and emits [batch, 6] by default, or per-token distributions with return_sequence=True. Supplying recurrent state is an advanced API: do not replay the same history with already-advanced state.

## Boundaries

This is a reduced-state predictive model, not a limit-order-book matching engine. Latent actions are not calibrated order volumes. External-feature ingestion is intentionally rejected by the packaged configuration until an aligned data adapter and evaluation exist.

The original notebook's recursive adapter does not reconstruct all 19 input features from six predicted targets. The packaged CLI therefore evaluates on observed input histories instead of exposing an unsupported autonomous rollout command.
