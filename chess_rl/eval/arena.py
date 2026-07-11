"""OnlineValuePolicy 체크포인트끼리 대국시켜 학습이 실제로 나아지고 있는지 상대적으로 평가.

절대적 기준(Stockfish 등) 대신, 시간 순으로 저장된 체크포인트들끼리 붙여서 "새
체크포인트가 과거 체크포인트 중 어디까지 이기는지"를 frontier로 추적한다
(docs/IDEAS.md '실력 측정 문제' 참고). 강도가 대체로 단조증가한다는 가정 하에,
전수/이분 탐색 대신 직전 새 체크포인트가 도달했던 frontier에서 시작해 이기면 위로,
지면 아래로 한 단계씩 옮겨가며 새 frontier를 찾는다 — 값이 크게 흔들리지 않는 한
매번 적은 매치 수만으로 충분하다.
"""

import chess

from chess_rl.rollout.game import play_game
from chess_rl.rollout.online_value_policy import OnlineValuePolicy
from chess_rl.utils.checkpoint import Checkpoint, load_checkpoint


class _StochasticPolicy:
    """OnlineValuePolicy.select_move를 항상 deterministic=False로 호출하는 얇은 wrapper.

    rollout.game.play_game()은 Policy.select_move(board)를 인자 없이 호출하므로,
    평가 대국에서 방문분포 샘플링(확률론적)을 쓰게 하려면 이렇게 감싸야 한다 — 안 그러면
    같은 두 체크포인트끼리 매번 완전히 똑같은 게임만 반복된다.
    """

    def __init__(self, policy: OnlineValuePolicy):
        self._policy = policy

    def select_move(self, board: chess.Board) -> chess.Move:
        return self._policy.select_move(board, deterministic=False)


def play_match(
    model_a,
    model_b,
    num_games: int = 100,
    mcts_simulations: int = 200,
    device: str = "cpu",
    max_moves: int = 300,
) -> dict:
    """model_a와 model_b를 num_games판 붙여 승/패/무 집계. 색은 절반씩 교대."""
    policy_a = _StochasticPolicy(OnlineValuePolicy(model_a, device=device, mcts_simulations=mcts_simulations))
    policy_b = _StochasticPolicy(OnlineValuePolicy(model_b, device=device, mcts_simulations=mcts_simulations))

    a_wins = b_wins = draws = 0
    a_games_as_white = num_games // 2
    for i in range(num_games):
        a_plays_white = i < a_games_as_white
        white, black = (policy_a, policy_b) if a_plays_white else (policy_b, policy_a)
        record = play_game(white, black, max_moves=max_moves)

        if record.result not in ("1-0", "0-1", "1/2-1/2", "*"):
            raise ValueError(f"unexpected game result {record.result!r}")

        # "*"(max_moves 안에 안 끝남, 예: threefold repetition을 아무도 claim 안 하고 계속
        # 버티는 경우)는 무승부로 adjudicate — engine 매치에서 흔한 관례.
        if record.result in ("1/2-1/2", "*"):
            draws += 1
        elif (record.result == "1-0") == a_plays_white:
            a_wins += 1
        else:
            b_wins += 1

    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws}


def a_beats_b(match: dict) -> bool:
    """draw를 0.5점으로 쳐서 점수 비교. 동점은 '못 이긴 것'으로 보수적 처리."""
    a_score = match["a_wins"] + 0.5 * match["draws"]
    b_score = match["b_wins"] + 0.5 * match["draws"]
    return a_score > b_score


def find_new_frontier(
    new_model,
    old_checkpoints: list[Checkpoint],
    start_idx: int,
    num_games: int = 100,
    mcts_simulations: int = 200,
    device: str = "cpu",
    max_moves: int = 300,
) -> dict:
    """new_model이 old_checkpoints(games_trained 오름차순) 중 어디까지 이기는지 추적.

    start_idx: 직전 새 체크포인트가 도달했던 frontier index(인덱스가 낮을수록 더 짧게
    학습된 old checkpoint). 아직 아무것도 평가한 적 없으면 -1을 넘기면 됨(가장 짧게
    학습된 것부터 시작).

    old_checkpoints는 보통 new_model과 **다른 family**(예: 순수 self-play 계보)의 체크포인트
    목록이다 — 그래야 "이 family의 n판째가 저 family의 어디까지 이기는지" 계보 간 비교가 된다.

    반환: {"frontier_idx": 새 frontier(-1이면 가장 약한 old checkpoint한테도 짐),
           "matches": 실제로 붙은 매치들의 로그(opponent_family, opponent_games_trained 포함)}.
    """
    if not old_checkpoints:
        return {"frontier_idx": -1, "matches": []}

    idx = min(max(start_idx, 0), len(old_checkpoints) - 1)
    matches = []

    def _play_against(i: int) -> dict:
        old_model = load_checkpoint(old_checkpoints[i].path, device)
        match = play_match(new_model, old_model, num_games, mcts_simulations, device, max_moves)
        matches.append(
            {
                "opponent_family": old_checkpoints[i].family,
                "opponent_games_trained": old_checkpoints[i].games_trained,
                **match,
            }
        )
        return match

    if a_beats_b(_play_against(idx)):
        while idx + 1 < len(old_checkpoints):
            if not a_beats_b(_play_against(idx + 1)):
                break
            idx += 1
    else:
        won = False
        while idx - 1 >= 0:
            idx -= 1
            if a_beats_b(_play_against(idx)):
                won = True
                break
        if not won:
            idx = -1

    return {"frontier_idx": idx, "matches": matches}
