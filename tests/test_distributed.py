"""Real two-process CPU collectives and gradient synchronization."""

from datetime import timedelta
import multiprocessing as mp
import time
import pytest
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from sams.config import Config
from sams.model import MarketSimulationModel
from sams.distributed import Context, EvaluationSampler


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        torch.manual_seed(42)
        model = MarketSimulationModel(Config.load("configs/smoke.yaml").model)
        wrapped = DistributedDataParallel(model)
        before = next(model.parameters()).detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        torch.manual_seed(100 + rank)
        result = wrapped(torch.randn(2, 8, 19))
        loss = result["state_mean"].square().mean() + result["state_log_std"].square().mean()
        loss.backward()
        optimizer.step()
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        peer = flat.clone()
        dist.broadcast(peer, src=0)
        torch.testing.assert_close(flat, peer, rtol=0, atol=0)
        assert not torch.equal(before, next(model.parameters()))
        context = Context(rank, 2, torch.device("cpu"))
        assert not context.all_finite(rank == 0)
        total = torch.tensor(sum(EvaluationSampler(range(7), rank, 2)))
        dist.all_reduce(total)
        assert total.item() == 21
        torch.save({"synchronized": True}, f"{output}/{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_two_process_gloo(tmp_path):
    context = mp.get_context("spawn")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes = [
        context.Process(target=worker, args=(rank, rendezvous, str(tmp_path))) for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 90
    try:
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
        assert all((tmp_path / f"{rank}.pt").exists() for rank in range(2))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
