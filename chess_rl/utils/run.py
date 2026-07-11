"""run 디렉토리 생성 + 재현성 메타데이터(config, git commit, 환경) 스냅샷."""

from datetime import datetime
from pathlib import Path

from chess_rl.config import ExperimentConfig
from chess_rl.utils.repro import (
    assert_clean_working_tree,
    capture_pip_freeze,
    get_git_commit_hash,
)


def create_run_dir(config: ExperimentConfig, base_dir="runs", allow_dirty: bool = False) -> Path:
    """base_dir/<timestamp>_<config.name>/ 아래 checkpoints, tensorboard, meta 디렉토리를 만들고
    meta에 config, git commit hash, pip freeze 결과를 저장한다."""
    assert_clean_working_tree(allow_dirty=allow_dirty)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{timestamp}_{config.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "tensorboard").mkdir()
    meta_dir = run_dir / "meta"
    meta_dir.mkdir()

    config.to_yaml(meta_dir / "config.yaml")
    (meta_dir / "git_commit.txt").write_text(get_git_commit_hash() + "\n")
    (meta_dir / "requirements_freeze.txt").write_text(capture_pip_freeze())

    return run_dir
