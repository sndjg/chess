"""온라인 학습되는 모델의 checkpoint 저장/조회/로드.

재현성 인프라(chess_rl.utils.run)와는 별개다 — OnlineValuePolicy는 이미 재현성을
의도적으로 포기했고(같은 조건이어도 판마다 다르게 학습됨), 여기서는 그 학습 궤적을
나중에(chess_rl.eval.arena) 서로 대국시켜 상대적으로 비교하기 위해 스냅샷만 남긴다.
파일명 규칙(game_{n:06d}.pt)으로 games_trained를 인코딩해서, 모델 전체를 pickle로
저장한다(이 프로젝트에서는 단순함을 우선; state_dict + 아키텍처 설정을 분리하는
방식보다 로드 쪽 코드가 가벼워짐).

**디렉터리 하나 = family(학습 계보) 하나**라는 관례를 쓴다 — 예: 사람과 함께 학습한
계보 vs 순수 self-play 계보처럼, 서로 다른 학습 방식/실행끼리 상대적으로 비교하는 게
목적이라(같은 계보 안에서의 진행 비교가 아님) games_trained만으로는 어느 계보 소속인지
구분이 안 된다. list_checkpoints()가 디렉터리 이름을 family로 간주해 Checkpoint에
채워 넣으므로, 결과를 나중에 봐도 어느 계보의 몇 판째인지 헷갈리지 않는다.

family 디렉터리에는 checkpoint 파일들과 함께 family_meta.json 하나를 둔다 — 이 family가
어떤 학습 방식(method, 자유 서술)으로, 어느 git commit에서, 언제 시작해서 마지막으로
언제까지 갱신됐는지 기록. "언제까지"는 OnlineValuePolicy가 서버 프로세스로 계속 살아있는
동안은 사실상 "마지막으로 checkpoint를 저장한 시각"이라 진행 중/중단됨을 구분하는
용도로도 쓸 수 있다(오래 안 갱신됐으면 사실상 끝난 것으로 간주).
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from chess_rl.utils.repro import get_git_commit_hash


@dataclass
class Checkpoint:
    family: str
    games_trained: int
    path: Path


@dataclass
class FamilyMeta:
    family: str
    method: str
    git_commit: str
    started_at: str
    last_updated_at: str


def save_checkpoint(model, checkpoint_dir: str, games_trained: int) -> Path:
    """checkpoint_dir는 family 하나에 대응하는 디렉터리여야 한다(예: checkpoints/online_value/human_online)."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"game_{games_trained:06d}.pt"
    torch.save(model, path)
    return path


def list_checkpoints(checkpoint_dir: str) -> list[Checkpoint]:
    """games_trained 오름차순으로 정렬된 checkpoint 목록. family는 디렉터리 이름으로 채움."""
    directory = Path(checkpoint_dir)
    if not directory.exists():
        return []

    family = directory.name
    checkpoints = []
    for path in directory.glob("game_*.pt"):
        games_trained = int(path.stem.removeprefix("game_"))
        checkpoints.append(
            Checkpoint(family=family, games_trained=games_trained, path=path)
        )
    checkpoints.sort(key=lambda c: c.games_trained)
    return checkpoints


def load_checkpoint(path, device: str = "cpu"):
    model = torch.load(path, map_location=device, weights_only=False)
    return model.to(device)


def _family_meta_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / "family_meta.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_family_meta(checkpoint_dir: str, family: str, method: str) -> FamilyMeta:
    """이 family를 위한 메타를 새로 만든다(한 번만 호출되는 것을 전제 — 이미 checkpoint가
    있는 family 디렉터리에서 새로 시작하려는 건 상위(OnlineValuePolicy)에서 막는다)."""
    now = _now_iso()
    meta = FamilyMeta(
        family=family,
        method=method,
        git_commit=get_git_commit_hash(),
        started_at=now,
        last_updated_at=now,
    )
    _save_family_meta(checkpoint_dir, meta)
    return meta


def touch_family_meta(checkpoint_dir: str) -> None:
    """checkpoint를 저장할 때마다 last_updated_at을 갱신 — '마지막으로 이 family가
    활동한 시각'을 남겨서, 나중에 봤을 때 진행 중인지 방치된 계보인지 가늠할 수 있게."""
    meta = read_family_meta(checkpoint_dir)
    meta.last_updated_at = _now_iso()
    _save_family_meta(checkpoint_dir, meta)


def read_family_meta(checkpoint_dir: str) -> FamilyMeta:
    data = json.loads(_family_meta_path(checkpoint_dir).read_text())
    return FamilyMeta(**data)


def _save_family_meta(checkpoint_dir: str, meta: FamilyMeta) -> None:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _family_meta_path(checkpoint_dir).write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False)
    )
