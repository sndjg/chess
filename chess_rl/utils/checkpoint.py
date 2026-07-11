"""온라인 학습되는 모델의 checkpoint 저장/조회/로드.

재현성 인프라(chess_rl.utils.run)와는 별개다 — OnlineValuePolicy는 이미 재현성을
의도적으로 포기했고(같은 조건이어도 판마다 다르게 학습됨), 여기서는 그 학습 궤적을
나중에(chess_rl.eval.arena) 서로 대국시켜 상대적으로 비교하기 위해 스냅샷만 남긴다.
파일명 규칙(game_{n:06d}.pt)으로 games_trained를 인코딩해서, 모델 전체를 pickle로
저장한다(이 프로젝트에서는 단순함을 우선; state_dict + 아키텍처 설정을 분리하는
방식보다 로드 쪽 코드가 가벼워짐).
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Checkpoint:
    games_trained: int
    path: Path


def save_checkpoint(model, checkpoint_dir: str, games_trained: int) -> Path:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"game_{games_trained:06d}.pt"
    torch.save(model, path)
    return path


def list_checkpoints(checkpoint_dir: str) -> list[Checkpoint]:
    """games_trained 오름차순으로 정렬된 checkpoint 목록."""
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return []

    checkpoints = []
    for path in directory.glob("game_*.pt"):
        games_trained = int(path.stem.removeprefix("game_"))
        checkpoints.append(Checkpoint(games_trained=games_trained, path=path))
    checkpoints.sort(key=lambda c: c.games_trained)
    return checkpoints


def load_checkpoint(path, device: str = "cpu"):
    model = torch.load(path, map_location=device, weights_only=False)
    return model.to(device)
