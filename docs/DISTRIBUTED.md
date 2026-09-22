# Distributed execution

SAMS uses data parallelism. Every rank owns the encoder and all agents. This implementation does not shard the agent population or model parameters.

The CLI reads torchrun rank variables, chooses Gloo for CPU and NCCL for CUDA, binds local GPU rank, synchronizes gradients through DDP, checks loss/gradient finiteness collectively, and writes artifacts only on rank zero.

Training drops the sampler tail and incomplete batches to give ranks equal work. Evaluation uses strided unpadded shards and reduces sums/counts once after local evaluation. A small dataset may have empty evaluation shards; those ranks still participate in reduction.

## Local smoke run

```bash
torchrun --standalone --nproc_per_node=2 -m sams.pretrain --config configs/smoke.yaml --data data/smoke.parquet --output runs/ddp --device cpu --max-steps 2
```

On Windows, the unit test uses a file rendezvous to avoid TCPStore/libuv build differences. The locally tested Windows PyTorch build rejects the standalone TCP rendezvous even with USE_LIBUV=0. Use the portable file-based launcher for a complete two-process CPU training smoke run:

```bash
python scripts/distributed_smoke.py --data data/smoke.parquet --output runs/ddp-local
```

It launches the same training module with two ranks, enforces a timeout and propagates worker failures. Linux is the CI target for the torchrun command. The file launcher is a local smoke utility, not a multi-node launcher.

## Multi-node launch template

On each node, set its node rank and use the same rendezvous address, run ID, configuration and data:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --rdzv_id=sams-run-001 --rdzv_backend=c10d --rdzv_endpoint=HOST:29500 -m sams.pretrain --config configs/pretrain.yaml --data /shared/optiver.parquet --output /shared/runs/pretrain --device cuda
```

Use node_rank=1 on the second node. This is a launch template, not a verified multi-node benchmark. Networking, CUDA/NCCL installation and matching data are the operator's responsibility. Use identical absolute shared paths for resume artifacts.

## What the automated check proves

[test_distributed.py](../tests/test_distributed.py) starts two real Gloo processes, runs different local inputs, backpropagates through the model, checks identical post-update parameters across ranks, confirms an actual parameter update, propagates a rank-local finite-check failure and verifies shard reduction.

It does not establish GPU speedup, elasticity, recovery from node loss or bitwise equivalence across hardware.
