import pytest
import torch
from sams.config import Config
from sams.synthetic import synthetic_frame


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def config():
    return Config.load("configs/smoke.yaml")


@pytest.fixture(scope="session")
def frame():
    return synthetic_frame()
