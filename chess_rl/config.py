"""실험 설정 dataclass + yaml 로드/저장.

MCTS/selfplay/train 모듈이 설계되면 관련 필드를 이 dataclass에 추가한다.
"""

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 0

    @classmethod
    def from_yaml(cls, path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(dataclasses.asdict(self), sort_keys=False))
