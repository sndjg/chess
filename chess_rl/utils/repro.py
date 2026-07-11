"""재현성 유틸: seed 고정, git 상태 확인, 환경(dependency) 스냅샷."""

import os
import random
import subprocess
import sys

import numpy as np
import torch


class DirtyWorkingTreeError(Exception):
    """git working tree에 uncommitted 변경사항이 있을 때 발생."""


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = False) -> None:
    """python/numpy/torch(+cuda) seed를 고정하고, cuDNN/cuBLAS의 비결정적 알고리즘 선택을 차단한다.

    CUBLAS_WORKSPACE_CONFIG는 첫 CUDA 연산 전에 설정돼야 cuBLAS가 읽으므로,
    학습 스크립트 맨 앞(어떤 텐서 연산보다도 먼저)에서 set_seed를 호출해야 한다.
    PYTHONHASHSEED는 프로세스 시작 전에 설정돼야 하는 값이라 여기서 되돌릴 수 없다 —
    필요하면 실행 전에 `PYTHONHASHSEED=<seed>`를 환경변수로 지정할 것.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


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
