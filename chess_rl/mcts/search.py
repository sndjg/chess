"""AlphaZero 스타일 MCTS: PUCT selection + network 기반 expansion/evaluation + backup.

학습 target 통합(policy/value 학습에 방문분포를 어떻게 쓸지, self-play 탐험을 위한
temperature/Dirichlet noise 등)은 이 모듈의 범위 밖이다. 여기서는 run()/run_batched()가
{move_uci: 방문 횟수} 분포와 root value만 반환한다.

run_batched()는 여러 독립적인 board(예: arena에서 동시에 진행 중인 여러 판)의 MCTS를
"시뮬레이션 단위로 lockstep" 진행시키면서, 매 시뮬레이션마다 각 board에서 나온 leaf를
모아 **한 번의 forward pass**로 network에 넣는다. Selection/backup은 board별 순수
Python 트리 순회라 배치화 이득이 없지만(애초에 병목이 아님), leaf 평가는 board 수만큼
개별 forward pass(batch size 1)를 GPU에 넣던 게 병목이었으므로 여기를 배치로 묶는 게
핵심. run(board, ...)은 run_batched([board], ...)의 결과 하나를 꺼내는 얇은 wrapper —
기존 단일 board 호출부(OnlineValuePolicy 등)는 변경 없이 그대로 쓸 수 있다.

board copy 비용 등 나머지 성능 항목은 docs/IDEAS.md 참고.
"""

import math

import chess
import numpy as np
import torch

from chess_rl.engine.action_space import MOVE_TO_INDEX
from chess_rl.engine.board import encode_board
from chess_rl.mcts.node import Node

C_PUCT = 1.5


def _puct_score(parent_visit_count: int, child: Node) -> float:
    """PUCT(Predictor + UCT) 점수 = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).

    - Q(s,a) = child.value: 이 수를 실제로 뒀을 때 지금까지 누적된 평균 value.
      "해보니 얼마나 좋았나"를 반영하는 활용(exploitation) 항.
    - P(s,a) = child.prior: network policy가 이 수에 준 사전 확률. 아직 해보지
      않았어도 network가 유망하다고 보는 정도.
    - sqrt(N(s)) / (1 + N(s,a)): 적게 방문한 수일수록 커지는 탐험(exploration)
      보너스. P(s,a)가 곱해져 있어서, network가 유망하다고 본 수는 덜 방문됐어도
      더 적극적으로 탐험된다. N(s)는 부모 국면의 총 방문 횟수(= 자식들 방문 횟수
      합, 부모 자신의 edge 통계는 없으므로 매번 자식들로부터 합산한다).
    - c_puct: 활용(Q)과 탐험(뒤 항) 사이 비중을 조절하는 상수.
    """
    exploration = (
        C_PUCT * child.prior * math.sqrt(parent_visit_count) / (1 + child.visit_count)
    )
    return child.value + exploration


def _select_child(node: Node) -> tuple[chess.Move, Node]:
    """Selection 단계: 자식들 중 PUCT 점수가 가장 높은 (수, 자식 노드)를 고른다."""
    parent_visit_count = sum(child.visit_count for child in node.children.values())
    return max(
        node.children.items(), key=lambda item: _puct_score(parent_visit_count, item[1])
    )


@torch.no_grad()
def _evaluate_batch(
    boards: list[chess.Board], model, device: str
) -> list[tuple[dict[chess.Move, float], float]]:
    """Expansion + Evaluation 단계: 여러 board를 한 번의 forward pass로 평가.

    반환값은 board마다 (합법수별 prior, board.turn 관점 value) — 두 값 모두 "지금 이
    국면에서 둘 차례인 쪽"의 관점이다(engine 전반의 관례와 동일, 예: OnlineValuePolicy).
    """
    planes = np.stack([encode_board(board) for board in boards])
    x = torch.from_numpy(planes).to(device)
    policy_logits, values = model(x)

    results = []
    for i, board in enumerate(boards):
        legal_moves = list(board.legal_moves)
        move_logits = np.array(
            [policy_logits[i, MOVE_TO_INDEX[move.uci()]].item() for move in legal_moves]
        )
        probs = np.exp(move_logits - move_logits.max())
        probs /= probs.sum()
        priors = {move: float(prob) for move, prob in zip(legal_moves, probs)}
        results.append((priors, values[i].item()))
    return results


def _terminal_value(board: chess.Board) -> float:
    """게임이 끝난 국면의 값을, board.turn(다음에 둘 차례인 쪽) 관점으로 반환."""
    result = board.result()
    if result == "1/2-1/2":
        return 0.0
    white_won = result == "1-0"
    return 1.0 if white_won == (board.turn == chess.WHITE) else -1.0


def run_batched(
    boards: list[chess.Board], model, num_simulations: int, device: str = "cpu"
) -> list[dict]:
    """여러 board에서 동시에 num_simulations번 MCTS 탐색 후, board별 방문분포/root value를 반환.

    board들은 서로 완전히 독립적인 게임이어도 된다(예: arena에서 동시에 진행 중인 여러
    판) — 매 시뮬레이션마다 모든 board의 selection을 먼저 끝내고, 그때 나온 leaf들을
    한 번의 batched forward pass로 같이 평가한 뒤 각자 backup한다("lockstep").

    반환: board마다 {"visit_counts": {move_uci: N(root, a)}, "root_value": ...} — run()과
    동일한 형식의 dict를 담은 리스트(입력 순서 유지).
    """
    roots = [Node() for _ in boards]
    root_evals = _evaluate_batch(boards, model, device)
    root_values = []
    for root, (priors, root_value) in zip(roots, root_evals):
        for move, prior in priors.items():
            root.children[move] = Node(prior=prior)
        root_values.append(root_value)

    for _ in range(num_simulations):
        sim_boards = [board.copy(stack=False) for board in boards]
        leaves = list(roots)
        paths = [[] for _ in boards]

        # Selection: 각 board를 독립적으로 leaf까지 내려간다(네트워크 호출 없는 순수 트리 순회).
        for g in range(len(boards)):
            node = roots[g]
            while node.children:
                move, node = _select_child(node)
                sim_boards[g].push(move)
                paths[g].append(node)
            leaves[g] = node

        # 종료 국면과 그렇지 않은 leaf를 나눠, 후자만 배치로 network 평가.
        values = [None] * len(boards)
        eval_indices = []
        eval_boards = []
        for g in range(len(boards)):
            if sim_boards[g].is_game_over():
                values[g] = _terminal_value(sim_boards[g])
            else:
                eval_indices.append(g)
                eval_boards.append(sim_boards[g])

        if eval_boards:
            batch_evals = _evaluate_batch(eval_boards, model, device)
            for g, (priors, value) in zip(eval_indices, batch_evals):
                for move, prior in priors.items():
                    leaves[g].children[move] = Node(prior=prior)
                values[g] = value

        # Backup: leaf에서 root 방향으로 값을 되돌린다. value는 leaf 국면에서
        # "다음에 둘 차례인 쪽"(= leaf로 들어온 수를 둔 사람의 상대) 관점이므로,
        # leaf 자신의 edge 통계(둔 사람 관점)에는 부호를 뒤집어 반영하고, 한 단계
        # 올라갈 때마다 관점이 다시 바뀌므로 부호도 다시 뒤집는다.
        for g in range(len(boards)):
            sign = -1
            for visited in reversed(paths[g]):
                visited.visit_count += 1
                visited.value_sum += sign * values[g]
                sign *= -1

    results = []
    for root, root_value in zip(roots, root_values):
        visit_counts = {
            move.uci(): child.visit_count for move, child in root.children.items()
        }
        results.append({"visit_counts": visit_counts, "root_value": root_value})
    return results


def run(board: chess.Board, model, num_simulations: int, device: str = "cpu") -> dict:
    """루트 board에서 num_simulations번 MCTS 탐색 후 방문분포와 root value를 반환.

    단일 board용 wrapper — run_batched([board], ...)의 결과 하나를 꺼내는 것과 동일.
    """
    return run_batched([board], model, num_simulations, device)[0]


def select_move_from_visit_counts(visit_counts: dict, deterministic: bool) -> str:
    """방문분포에서 실제로 둘 수를 고른다.

    deterministic=True: 방문 횟수 argmax. deterministic=False: 방문 횟수 비례 샘플링 —
    같은 두 정책끼리 반복 대국시켜도 매번 다른 게임이 나오게 하기 위함(예: arena 평가).
    """
    if deterministic:
        return max(visit_counts, key=visit_counts.get)

    ucis = list(visit_counts.keys())
    counts = np.array([visit_counts[uci] for uci in ucis], dtype=np.float64)
    probs = counts / counts.sum()
    return np.random.choice(ucis, p=probs)
