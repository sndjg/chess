"""재현성 인프라(run 디렉토리 생성 + 메타데이터 스냅샷 + tensorboard 로깅) 스모크 테스트.

mcts/selfplay/train이 아직 없으므로, 더미 스칼라 하나를 tensorboard에 기록해
인프라 자체가 end-to-end로 동작하는지만 확인한다.
"""

import argparse

from torch.utils.tensorboard import SummaryWriter

from chess_rl.config import ExperimentConfig
from chess_rl.utils.repro import set_seed
from chess_rl.utils.run import create_run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="chess_rl/configs/default.yaml")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.seed)
    run_dir = create_run_dir(config, allow_dirty=args.allow_dirty)

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    writer.add_scalar("smoke/dummy_metric", 1.0, global_step=0)
    writer.close()

    print(f"run_dir: {run_dir}")
