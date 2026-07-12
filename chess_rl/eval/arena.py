"""OnlineValuePolicy 체크포인트끼리 대국시켜 학습이 실제로 나아지고 있는지 상대적으로 평가.

절대적 기준(Stockfish 등) 대신, 시간 순으로 저장된 체크포인트들끼리 붙여서 "새
체크포인트가 과거 체크포인트 중 어디까지 이기는지"를 frontier로 추적한다
(docs/IDEAS.md '실력 측정 문제' 참고). 강도가 대체로 단조증가한다는 가정 하에,
전수/이분 탐색 대신 직전 새 체크포인트가 도달했던 frontier에서 시작해 이기면 위로,
지면 아래로 한 단계씩 옮겨가며 새 frontier를 찾는다 — 값이 크게 흔들리지 않는 한
매번 적은 매치 수만으로 충분하다.

play_match()는 num_games판을 순차로 하나씩 두지 않고, 모든 판을 동시에 "한 수씩"
진행시키면서 그 라운드에 같은 model이 둘 차례인 판들을 모아 mcts.search.run_batched()로
한 번에 평가한다 — 매 수마다 batch size 1짜리 forward pass를 개별로 GPU에 넣는 게
병목이었기 때문에, 여러 판의 leaf를 배치로 묶어 forward pass 횟수 자체를 줄인다.
"""

import time

import chess

from chess_rl.mcts.search import MOVE_SELECTORS, run_batched
from chess_rl.utils.checkpoint import Checkpoint, load_checkpoint


def play_match(
    model_a,
    model_b,
    num_games: int = 100,
    mcts_simulations: int = 200,
    device: str = "cpu",
    max_moves: int = 300,
    selector_a: str = "visits",
    selector_b: str = "visits",
) -> dict:
    """model_a와 model_b를 num_games판 붙여 승/패/무 집계. 색은 절반씩 교대.

    num_games판을 동시에 진행시키는 배치 드라이버 — 각 라운드마다 아직 안 끝난 판들을
    "이번 차례가 model_a인 판"과 "model_b인 판"으로 나눠 run_batched()를 한 번씩만 호출.

    selector_a/selector_b: 탐색 결과에서 수를 고르는 전략 이름(mcts.search.MOVE_SELECTORS).
    같은 model을 서로 다른 선택 전략으로 붙여서 전략 자체를 실측 비교할 수도 있다.
    """
    select_a = MOVE_SELECTORS[selector_a]
    select_b = MOVE_SELECTORS[selector_b]
    boards = [chess.Board() for _ in range(num_games)]
    ply_counts = [0] * num_games
    a_games_as_white = num_games // 2
    a_is_white = [i < a_games_as_white for i in range(num_games)]

    def active_indices() -> list[int]:
        return [
            i
            for i in range(num_games)
            if not boards[i].is_game_over() and ply_counts[i] < max_moves
        ]

    def mover_is_a(i: int) -> bool:
        return (boards[i].turn == chess.WHITE) == a_is_white[i]

    while True:
        active = active_indices()
        if not active:
            break

        # 두 그룹 다 이번 라운드 시작 시점의 board 상태로 한 번에 나눈다 — model_a
        # 그룹을 먼저 처리하면서 수를 두면 차례(turn)가 바뀌므로, model_b 그룹을 그
        # *다음에* 다시 계산하면 방금 움직인 board가 turn이 바뀌어 엉뚱하게 두 번째
        # 그룹에도 끼어들어 한 라운드에 같은 board가 두 번 움직이는 버그가 생긴다.
        group_a = [i for i in active if mover_is_a(i)]
        group_b = [i for i in active if not mover_is_a(i)]

        for model, group, select in (
            (model_a, group_a, select_a),
            (model_b, group_b, select_b),
        ):
            if not group:
                continue

            results = run_batched(
                [boards[i] for i in group], model, mcts_simulations, device
            )
            for i, result in zip(group, results):
                uci = select(result, False)  # deterministic=False: 게임 다양성 확보
                boards[i].push(chess.Move.from_uci(uci))
                ply_counts[i] += 1

    a_wins = b_wins = draws = 0
    for i in range(num_games):
        result = boards[i].result() if boards[i].is_game_over() else "*"

        if result not in ("1-0", "0-1", "1/2-1/2", "*"):
            raise ValueError(f"unexpected game result {result!r}")

        # "*"(max_moves 안에 안 끝남, 예: threefold repetition을 아무도 claim 안 하고 계속
        # 버티는 경우)는 무승부로 adjudicate — engine 매치에서 흔한 관례.
        if result in ("1/2-1/2", "*"):
            draws += 1
        elif (result == "1-0") == a_is_white[i]:
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
    on_match=None,
) -> dict:
    """new_model이 old_checkpoints(games_trained 오름차순) 중 어디까지 이기는지 추적.

    start_idx: 직전 새 체크포인트가 도달했던 frontier index(인덱스가 낮을수록 더 짧게
    학습된 old checkpoint). 아직 아무것도 평가한 적 없으면 -1을 넘기면 됨(가장 짧게
    학습된 것부터 시작).

    old_checkpoints는 보통 new_model과 **다른 family**(예: 순수 self-play 계보)의 체크포인트
    목록이다 — 그래야 "이 family의 n판째가 저 family의 어디까지 이기는지" 계보 간 비교가 된다.

    on_match(match_entry, won): 주어지면 매치 하나가 끝날 때마다(전체 walk이 끝나기 전에도)
    호출된다 — walk 하나에 여러 매치가 걸릴 수 있으니, 매치 단위로 진행 상황을 바로바로
    UI에 반영하고 싶을 때 씀(예: viz 서버의 실시간 비교 패널).

    반환: {"frontier_idx": 새 frontier(-1이면 가장 약한 old checkpoint한테도 짐),
           "matches": 실제로 붙은 매치들의 로그(opponent_family, opponent_games_trained 포함)}.
    """
    if not old_checkpoints:
        return {"frontier_idx": -1, "matches": []}

    idx = min(max(start_idx, 0), len(old_checkpoints) - 1)
    matches = []

    def _play_against(i: int) -> bool:
        old_model = load_checkpoint(old_checkpoints[i].path, device)
        start = time.time()
        match = play_match(
            new_model, old_model, num_games, mcts_simulations, device, max_moves
        )
        elapsed_seconds = time.time() - start
        match_entry = {
            "opponent_family": old_checkpoints[i].family,
            "opponent_games_trained": old_checkpoints[i].games_trained,
            "elapsed_seconds": elapsed_seconds,
            **match,
        }
        matches.append(match_entry)
        won = a_beats_b(match)
        if on_match is not None:
            on_match(match_entry, won)
        return won

    if _play_against(idx):
        while idx + 1 < len(old_checkpoints):
            if not _play_against(idx + 1):
                break
            idx += 1
    else:
        won_any = False
        while idx - 1 >= 0:
            idx -= 1
            if _play_against(idx):
                won_any = True
                break
        if not won_any:
            idx = -1

    return {"frontier_idx": idx, "matches": matches}
