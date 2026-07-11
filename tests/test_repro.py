import os
import shutil

import torch

from chess_rl.config import ExperimentConfig
from chess_rl.utils.repro import get_git_commit_hash, set_seed
from chess_rl.utils.run import create_run_dir


def test_set_seed_determinism():
    set_seed(0)
    a = torch.rand(4)
    set_seed(0)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_set_seed_enables_deterministic_algorithms():
    set_seed(0)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_set_seed_conv_backward_reproducible():
    """cudnn.deterministic/use_deterministic_algorithms가 실제로 conv backward에도 적용되는지 확인."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def run():
        set_seed(0)
        x = torch.randn(2, 3, 8, 8, device=device, requires_grad=True)
        conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1).to(device)
        out = conv(x)
        out.sum().backward()
        assert x.grad is not None
        assert conv.weight.grad is not None
        return x.grad.clone(), conv.weight.grad.clone()

    x_grad_a, w_grad_a = run()
    x_grad_b, w_grad_b = run()
    assert torch.equal(x_grad_a, x_grad_b)
    assert torch.equal(w_grad_a, w_grad_b)


def test_get_git_commit_hash_is_full_sha():
    commit_hash = get_git_commit_hash()
    assert len(commit_hash) == 40
    int(commit_hash, 16)  # hex string이어야 함


def test_create_run_dir_writes_meta(tmp_path):
    config = ExperimentConfig(name="pytest_smoke", seed=42)
    run_dir = create_run_dir(config, base_dir=tmp_path, allow_dirty=True)

    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "tensorboard").is_dir()
    assert (run_dir / "meta" / "config.yaml").exists()
    assert (run_dir / "meta" / "git_commit.txt").exists()
    assert (run_dir / "meta" / "requirements_freeze.txt").exists()

    loaded = ExperimentConfig.from_yaml(run_dir / "meta" / "config.yaml")
    assert loaded == config

    shutil.rmtree(run_dir)
