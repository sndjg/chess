"""판마다 실제 대국 결과로 policy/value head를 함께 학습하는 온라인 정책.

재현성 인프라(ExperimentConfig, seed 고정, run 디렉토리)는 의도적으로 쓰지 않는다 —
사람과의 실시간 대국에 맞춰 서버 프로세스 동안 계속 다르게 학습되는 게 이 기능의 목적이라
재현성을 포기함(서버 재시작 시 초기화).

value head: Monte-Carlo 회귀(그 국면 차례 관점의 최종 결과로 MSE).
policy head: 게임에 나온 모든 수(사람 몫 포함)를 value-delta 가중으로 강화 —
매 epoch log-prob(실제 둔 수) * (그 epoch에 value 예측이 움직인 양). REINFORCE의
advantage 가중을 epoch별로 잘게 나눈 변형(총합이 telescoping으로 advantage에 수렴)으로,
같은 배치 다스텝 학습에서 REINFORCE가 발산하는 문제를 피하기 위한 비표준 기법 —
근거와 위험은 docs/IDEAS.md 'value-delta 가중 policy 학습' 참고. 사람이 둔 수도
포함하므로 "이긴 판의 수는 흉내내고 진 판의 수는 피하는" 결과-가중 모방 학습 성격.

실제 수 선택(select_move)은 policy head를 직접 쓰지 않고 chess_rl.mcts.search.run()으로
매 수마다 MCTS 탐색을 돌린다(추론에도 MCTS를 포함시킨 버전 — 지연이 느껴지면
mcts_simulations를 낮추는 방향으로 조정 예정). 탐색 후 수 선택은 두 가지 모드:
deterministic=True(기본값, 사람과의 실제 대국)는 방문 횟수 argmax, deterministic=False
(체크포인트 간 평가 대국 등)는 방문 횟수 분포에서 샘플링 — 후자는 같은 두 정책끼리 여러
판 반복 대국시켜도 매번 다른 게임이 나오게 하기 위한 것. MCTS 탐색 target 통합(policy
head를 방문분포로 학습시키는 것 등)은 아직 안 함 — policy/value head 학습은 여전히
기존 REINFORCE/MSE 방식 그대로.

판이 끝나면 그 판의 포지션들을 ReplayBuffer(chess_rl.rollout.replay_buffer)에 쌓아두고,
그 판 데이터만으로 학습하는 대신 buffer에서 배치를 무작위로 샘플링해 학습한다 — 판 하나
분량으로 학습이 캡되는 문제를 완화하기 위함.

checkpoint_dir이 주어지면(family와 함께, 필수) checkpoint_every판마다
checkpoint_dir/family/ 아래에 모델 스냅샷을 저장한다(chess_rl.utils.checkpoint) —
chess_rl.eval.arena가 이 스냅샷들끼리(다른 family와도) 대국시켜 학습이 실제로
나아지고 있는지 상대적으로 평가하는 데 쓴다.

동시성: OnlineValuePolicy는 "canonical" 학습 상태(모델/optimizer/replay buffer)를 들고
있는 트레이너 역할이다. 실제 대국(추론)에는 이 객체를 직접 쓰지 않고 new_inference_handle()로
얻은 _InferenceHandle을 쓴다 — learn_from_game()이 train_epochs만큼 순전파/역전파를 도는
동안(오래 걸릴 수 있음) 다른 대국이 계속 추론하려면 같은 model 객체를 공유하면 안 되기
때문(model.train()/eval() 모드 충돌, 학습 중인 가중치로 forward하는 레이스 등 — 두 스레드가
같은 nn.Module을 동시에 건드리는 문제). 핸들은 대국 시작 시 canonical 가중치를 스냅샷 떠서
만들고 그 대국 내내 그대로 쓰며, learn_from_game은 canonical의 별도 복사본에서 학습을
진행한 뒤 끝나면 canonical에 병합한다 — lock은 복사(스냅샷)와 병합, 두 짧은 구간에서만 잡고
무거운 학습 자체는 lock 밖에서 진행한다. optimizer(Adam) momentum도 복사·병합 대상이라
학습 품질(모멘텀 연속성)이 유지된다.

학습끼리의 겹침: 학습이 진행 중일 때 또 다른 판이 끝나면, 진행 중인 학습에 조기 종료를
요청하고(대기자 카운터를 학습 루프가 매 epoch 확인) 그쪽이 진행분을 canonical에 병합하고
나온 뒤, 병합된 canonical + 새 판까지 포함된 buffer에서 새 학습을 시작한다(_train_lock으로
직렬화). 병렬로 돌려서 늦게 끝난 쪽이 먼저 끝난 쪽을 덮어쓰는(last-writer-wins) 방식보다
먼저 판의 진행분이 보존되고, value-delta 방식의 유계성 덕에 어느 시점에 끊겨도 안전하다.
"""

import copy
import threading

from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from chess_rl.engine.action_space import MOVE_TO_INDEX
from chess_rl.engine.board import (
    encode_board,
    legal_move_mask,
    terminal_value_for_side_to_move,
)
from chess_rl.mcts.search import MOVE_SELECTORS, select_by_visit_count
from chess_rl.mcts.search import run as mcts_run
from chess_rl.rollout.replay_buffer import ReplayBuffer
from chess_rl.utils.checkpoint import (
    list_checkpoints,
    save_checkpoint,
    touch_family_meta,
    write_family_meta,
)


def _result_to_white_score(result: str) -> float:
    return {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(result, 0.0)


def _select_move(
    model,
    board: chess.Board,
    mcts_simulations: int,
    device: str,
    deterministic: bool,
    selector=None,
) -> chess.Move:
    model.eval()
    result = mcts_run(board, model, num_simulations=mcts_simulations, device=device)
    selector = selector or select_by_visit_count
    return chess.Move.from_uci(selector(result, deterministic))


def _search_move_with_candidates(
    model,
    board: chess.Board,
    mcts_simulations: int,
    device: str,
    deterministic: bool,
    selector=None,
) -> tuple[chess.Move, list]:
    """MCTS를 한 번만 돌려서 (선택된 수, 후보 수 목록)을 함께 반환.

    후보 목록은 같은 탐색의 root 통계에서 나온다 — value는 Q(그 수를 두는 쪽 관점,
    탐색 누적 평균), visits는 방문 횟수. 화살표 시각화용 평가와 실제 수 선택의 근거가
    같은 탐색이 되도록(따로 계산하면 서로 다른 걸 보여주게 됨) 만든 함수."""
    model.eval()
    result = mcts_run(board, model, num_simulations=mcts_simulations, device=device)
    candidates = [
        {
            "move": uci,
            "value": result["root_q"][uci],
            "visits": result["visit_counts"][uci],
        }
        for uci in result["visit_counts"]
    ]
    # 방문 횟수 우선 정렬(동률이면 Q) — MCTS의 결론은 방문분포이고, 방문 0인 수는 Q=0이라
    # value 정렬로는 "검토조차 안 한 수"가 음수 Q인 검토된 수들 위로 올라오는 왜곡이 생김.
    candidates.sort(key=lambda c: (c["visits"], c["value"]), reverse=True)
    selector = selector or select_by_visit_count
    return chess.Move.from_uci(selector(result, deterministic)), candidates


def _forward(model, board: chess.Board, device: str):
    planes = encode_board(board)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    return model(x)


@torch.no_grad()
def _value_estimate_white_perspective(model, board: chess.Board, device: str) -> float:
    # 종료된 국면(체크메이트 등)은 network가 학습해본 적 없는 입력이라 raw forward
    # pass로 평가하면 신뢰할 수 없음 — 실제 결과를 직접 씀(mcts와 동일한 처리).
    if board.is_game_over():
        value = terminal_value_for_side_to_move(board)
    else:
        model.eval()
        _, raw_value = _forward(model, board, device)
        value = raw_value.item()
    return value if board.turn == chess.WHITE else -value


@torch.no_grad()
def _move_values(model, board: chess.Board, device: str) -> list:
    """현재 board에서 둘 수 있는 모든 수를, 그 수를 두는 쪽 관점의 가치로 평가해 내림차순 정렬."""
    model.eval()
    results = []
    for move in board.legal_moves:
        next_board = board.copy(stack=False)
        next_board.push(move)
        if next_board.is_game_over():
            value = terminal_value_for_side_to_move(next_board)
        else:
            _, raw_value = _forward(model, next_board, device)
            value = raw_value.item()
        value_for_mover = -value  # next_board는 상대 관점이라 부호 반전
        results.append({"move": move.uci(), "value": value_for_mover})
    results.sort(key=lambda r: r["value"], reverse=True)
    return results


class OnlineValuePolicy:
    """canonical 학습 상태(모델/optimizer/replay buffer) 보유자.

    실제 대국은 new_inference_handle()로 얻은 독립 복사본(_InferenceHandle)이 담당 —
    모듈 docstring '동시성' 절 참고.
    """

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        train_epochs: int = 5,
        device: str = "cpu",
        mcts_simulations: int = 200,
        replay_capacity: int = 5000,
        batch_size: int = 256,
        checkpoint_dir: str | None = None,
        family: str | None = None,
        training_method: str | None = None,
        checkpoint_every: int = 1,
        move_selector: str = "visits",
    ):
        if checkpoint_dir is not None:
            if not family:
                raise ValueError(
                    "checkpoint_dir가 주어지면 family(학습 계보 식별자)도 함께 줘야 함"
                )
            if not training_method:
                raise ValueError(
                    "checkpoint_dir가 주어지면 training_method(학습 방식 서술)도 함께 줘야 함"
                )
        if move_selector not in MOVE_SELECTORS:
            raise ValueError(
                f"unknown move_selector '{move_selector}' — 사용 가능: {sorted(MOVE_SELECTORS)}"
            )

        # 탐색 결과에서 실제 둘 수를 고르는 전략(mcts.search.MOVE_SELECTORS에 병치된
        # 것들 중 이름으로 선택) — 전략끼리 arena로 실측 비교/교체하기 쉽게 하기 위함.
        self.move_selector = move_selector
        self._selector_fn = MOVE_SELECTORS[move_selector]

        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_epochs = train_epochs
        self.games_trained = 0
        # 누적 학습 epoch — 조기 중단(양보) 때문에 판마다 실제 학습량이 달라서, 계보 간
        # 비교의 축으로는 games_trained보다 이 값이 정확하다.
        self.total_epochs_trained = 0
        self.mcts_simulations = mcts_simulations
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)
        self.batch_size = batch_size
        self.family = family
        self.checkpoint_every = checkpoint_every
        self._lock = threading.Lock()
        # 학습 직렬화용: 학습 전체 구간(스냅샷~병합)을 하나만 진행되게 함. 새 학습이
        # 대기 중이면(_waiting_trainings > 0) 진행 중인 학습이 조기 종료하고 병합 후
        # 양보한다 — learn_from_game() 참고.
        self._train_lock = threading.Lock()
        self._waiting_trainings = 0
        # canonical이 병합으로 갱신될 때마다 +1 — 핸들이 "내 스냅샷이 낡았나"를 싸게
        # 확인하고 필요할 때만 다시 복사하는(lazy refresh) 데 쓴다.
        self._model_version = 0

        if checkpoint_dir is not None:
            family_dir = str(Path(checkpoint_dir) / family)
            if list_checkpoints(family_dir):
                raise ValueError(
                    f"family '{family}' 디렉터리({family_dir})에 이미 checkpoint가 있음 — "
                    "같은 family를 재사용하면 games_trained가 1부터 다시 시작되어 기존 파일을 "
                    "덮어쓰게 됨. 새 family 이름을 쓸 것."
                )
            write_family_meta(family_dir, family, training_method)
            self.checkpoint_dir = family_dir
        else:
            self.checkpoint_dir = None

    def select_move(self, board: chess.Board, deterministic: bool = True) -> chess.Move:
        """deterministic=True(기본값, 사람과의 실제 대국용): move_selector 전략의 argmax.
        deterministic=False(체크포인트 간 평가 대국 등): 전략별 샘플링 — 같은 두 정책끼리
        반복 대국시켜도 매번 다른 게임이 나오게 하기 위함."""
        return _select_move(
            self.model,
            board,
            self.mcts_simulations,
            self.device,
            deterministic,
            selector=self._selector_fn,
        )

    def search_move_with_candidates(
        self, board: chess.Board, deterministic: bool = True
    ) -> tuple[chess.Move, list]:
        """MCTS를 한 번 돌려 (선택된 수, root 후보 통계)를 함께 반환 — 화살표 시각화가
        실제 수 선택과 같은 탐색을 근거로 삼게 하기 위함."""
        return _search_move_with_candidates(
            self.model,
            board,
            self.mcts_simulations,
            self.device,
            deterministic,
            selector=self._selector_fn,
        )

    def value_estimate_white_perspective(self, board: chess.Board) -> float:
        return _value_estimate_white_perspective(self.model, board, self.device)

    def move_values(self, board: chess.Board) -> list:
        return _move_values(self.model, board, self.device)

    def new_inference_handle(self) -> "_InferenceHandle":
        """대국 세션 하나가 쓸 canonical 모델의 독립 복사본을 반환.

        학습(learn_from_game)이 오래 걸리는 동안에도(train_epochs가 크면 특히) 그 학습과
        무관하게 안전하게 추론할 수 있다 — 모듈 docstring '동시성' 절 참고. 핸들은 사용
        시점마다 canonical 버전을 확인해서, 백그라운드 학습으로 canonical이 갱신됐으면
        다음 수 계산 전에 최신 복사본으로 갈아탄다(lazy refresh) — 대국 도중에도 학습
        성과가 즉시 반영됨.
        """
        with self._lock:
            model_copy = copy.deepcopy(self.model)
            version = self._model_version
        return _InferenceHandle(model_copy, self, version)

    def learn_from_game(self, moves: list, result: str) -> dict:
        """한 판(moves, result)을 replay buffer에 적립하고, canonical의 복사본에서
        학습을 진행한 뒤 끝나면 canonical에 병합한다(모델 + optimizer state)."""
        board = chess.Board()
        states = []
        value_targets = []
        action_indices = []
        masks = []

        white_score = _result_to_white_score(result)
        for move_uci in moves:
            target = white_score if board.turn == chess.WHITE else -white_score
            states.append(encode_board(board))
            value_targets.append(target)
            action_indices.append(MOVE_TO_INDEX[move_uci])
            masks.append(legal_move_mask(board, MOVE_TO_INDEX))
            board.push(chess.Move.from_uci(move_uci))

        # 학습이 이미 진행 중인데 또 다른 판이 끝난 경우: 진행 중인 학습에게 조기 종료를
        # 요청하고(대기자 카운터), 그쪽이 지금까지의 진행분을 canonical에 병합하고 나올
        # 때까지 기다렸다가(_train_lock), 병합된 최신 canonical + 방금 판까지 포함된
        # buffer에서 새로 학습을 시작한다 — 겹친 학습이 서로를 덮어써서 한쪽 결과가
        # 통째로 유실되던 문제(last-writer-wins)의 해결.
        with self._lock:
            self.replay_buffer.add_game(states, value_targets, action_indices, masks)
            self._waiting_trainings += 1

        with self._train_lock:
            with self._lock:
                self._waiting_trainings -= 1
                model_copy = copy.deepcopy(self.model)
                optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=self.lr)
                optimizer_copy.load_state_dict(
                    copy.deepcopy(self.optimizer.state_dict())
                )
                batch_states, batch_value_targets, batch_action_indices, batch_masks = (
                    self.replay_buffer.sample(self.batch_size)
                )

            # 무거운 학습(순전파/역전파 * train_epochs)은 복사본에 대해서만, canonical
            # lock(self._lock) 밖에서 진행 — 그동안 다른 대국은 canonical(또는 각자의
            # 핸들 복사본)로 자유롭게 계속 추론할 수 있다.
            x = torch.from_numpy(batch_states).to(self.device)
            y = torch.from_numpy(batch_value_targets).to(self.device)
            action_idx = torch.from_numpy(batch_action_indices).to(self.device)
            mask = torch.from_numpy(batch_masks).to(self.device)
            num_positions = len(batch_states)

            model_copy.eval()
            with torch.no_grad():
                _, pred_before = model_copy(x)
                loss_before = F.mse_loss(pred_before, y).item()

            # policy는 value-delta 가중 방식으로 학습한다(비표준 기법 — docs/IDEAS.md
            # 'value-delta 가중 policy 학습' 참고). 기존 REINFORCE(advantage = y - pred)는
            # 같은 배치로 다스텝을 돌면 loss가 아래로 비유계라 발산 — 실측으로 value head가
            # 포화되어 죽는 것까지 확인됨. 대신 매 epoch "value 예측이 이번에 움직인 양"
            # (pred_t - pred_{t-1})을 가중치로 쓰면 epoch별 delta의 총합이 telescoping으로
            # advantage에 수렴해 총 업데이트량이 유계이고, value가 수렴/정체하면 delta -> 0으로
            # policy도 자연 감쇠한다. 이 유계성 덕에 아래의 "대기자 생기면 조기 중단"도
            # 안전하다 — 어느 시점에 끊겨도 policy 업데이트 총량이 그때까지의 value 변화량
            # 이하로 제한됨.
            model_copy.train()
            prev_pred = None
            epochs_run = 0
            interrupted = False
            for _ in range(self.train_epochs):
                # 다른 판의 학습이 대기 중이면 조기 종료하고 진행분을 병합 — CPython에서
                # int 읽기는 원자적이라 lock 없이 확인해도 안전(한 epoch 늦게 보는 정도).
                if self._waiting_trainings > 0:
                    interrupted = True
                    break
                optimizer_copy.zero_grad()
                policy_logits, pred = model_copy(x)
                value_loss = F.mse_loss(pred, y)

                loss = value_loss
                if prev_pred is not None:
                    masked_logits = policy_logits.masked_fill(mask == 0, float("-inf"))
                    log_probs = F.log_softmax(masked_logits, dim=-1)
                    selected_log_probs = log_probs[
                        torch.arange(num_positions), action_idx
                    ]
                    delta = (pred - prev_pred).detach()
                    policy_loss = -(selected_log_probs * delta).mean()
                    loss = loss + policy_loss
                prev_pred = pred.detach()

                loss.backward()
                optimizer_copy.step()
                epochs_run += 1
            model_copy.eval()

            with torch.no_grad():
                _, pred_after = model_copy(x)
                loss_after = F.mse_loss(pred_after, y).item()

            with self._lock:
                self.model.load_state_dict(model_copy.state_dict())
                self.optimizer.load_state_dict(optimizer_copy.state_dict())
                self.games_trained += 1
                self.total_epochs_trained += epochs_run
                self._model_version += 1

                checkpoint_path = None
                if (
                    self.checkpoint_dir is not None
                    and self.games_trained % self.checkpoint_every == 0
                ):
                    checkpoint_path = save_checkpoint(
                        self.model,
                        self.checkpoint_dir,
                        self.games_trained,
                        total_epochs=self.total_epochs_trained,
                    )
                    touch_family_meta(self.checkpoint_dir)
                games_trained = self.games_trained
                total_epochs_trained = self.total_epochs_trained
                buffer_size = len(self.replay_buffer)

        return {
            "num_positions": num_positions,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "games_trained": games_trained,
            "buffer_size": buffer_size,
            "epochs_run": epochs_run,
            "total_epochs_trained": total_epochs_trained,
            "interrupted": interrupted,
            "checkpoint_path": str(checkpoint_path)
            if checkpoint_path is not None
            else None,
        }

    def _forward(self, board: chess.Board):
        return _forward(self.model, board, self.device)


class _InferenceHandle:
    """대국 세션 하나가 쓰는, canonical과 독립된 model 복사본 기반 추론 전용 정책.

    학습은 이 복사본이 아니라 트레이너(OnlineValuePolicy)에서 진행되고 결과가 나중에
    canonical에 병합된다 — OnlineValuePolicy.new_inference_handle() 참고. canonical이
    갱신되면 다음 사용 시점에 최신 복사본으로 갈아탄다(_refresh_if_stale).
    """

    def __init__(self, model, trainer: OnlineValuePolicy, model_version: int):
        self.model = model
        self._trainer = trainer
        self._model_version = model_version

    def _refresh_if_stale(self) -> None:
        """canonical이 백그라운드 학습으로 갱신됐으면 최신 복사본으로 갈아탄다.

        버전 비교(정수 읽기)는 싸서 매 사용 시 확인해도 부담 없고, 실제 deepcopy는
        버전이 바뀐 경우에만 일어난다. 이 덕에 대국 도중에도 직전 판(들)의 학습 성과가
        다음 수부터 바로 반영된다.
        """
        if self._model_version == self._trainer._model_version:
            return
        with self._trainer._lock:
            self.model = copy.deepcopy(self._trainer.model)
            self._model_version = self._trainer._model_version

    def select_move(self, board: chess.Board, deterministic: bool = True) -> chess.Move:
        self._refresh_if_stale()
        return _select_move(
            self.model,
            board,
            self._trainer.mcts_simulations,
            self._trainer.device,
            deterministic,
            selector=self._trainer._selector_fn,
        )

    def search_move_with_candidates(
        self, board: chess.Board, deterministic: bool = True
    ) -> tuple[chess.Move, list]:
        self._refresh_if_stale()
        return _search_move_with_candidates(
            self.model,
            board,
            self._trainer.mcts_simulations,
            self._trainer.device,
            deterministic,
            selector=self._trainer._selector_fn,
        )

    def value_estimate_white_perspective(self, board: chess.Board) -> float:
        self._refresh_if_stale()
        return _value_estimate_white_perspective(
            self.model, board, self._trainer.device
        )

    def move_values(self, board: chess.Board) -> list:
        self._refresh_if_stale()
        return _move_values(self.model, board, self._trainer.device)

    def learn_from_game(self, moves: list, result: str) -> dict:
        return self._trainer.learn_from_game(moves, result)

    @property
    def games_trained(self) -> int:
        return self._trainer.games_trained
