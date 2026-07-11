"""재현성 유틸: seed 고정, git 상태 확인, 환경(dependency) 스냅샷."""

import random
import subprocess
import sys

import numpy as np
import torch


class DirtyWorkingTreeError(Exception):
    """git working tree에 uncommitted 변경사항이 있을 때 발생."""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def is_working_tree_dirty() -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    return bool(status.strip())


def assert_clean_working_tree(allow_dirty: bool = False) -> None:
    if is_working_tree_dirty() and not allow_dirty:
        raise DirtyWorkingTreeError(
            "git working tree에 uncommitted 변경사항이 있습니다. "
            "커밋 후 다시 실행하거나 allow_dirty=True로 명시적으로 허용하세요."
        )


def capture_pip_freeze() -> str:
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
