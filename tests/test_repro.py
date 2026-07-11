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
