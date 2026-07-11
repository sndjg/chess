"""AlphaZero 스타일 MCTS: PUCT selection + network 기반 expansion/evaluation + backup.

학습 target 통합(policy/value 학습에 방문분포를 어떻게 쓸지, self-play 탐험을 위한
temperature/Dirichlet noise 등)은 이 모듈의 범위 밖이다. 여기서는 run()이
{move_uci: 방문 횟수} 분포와 root value만 반환한다.

성능 최적화(leaf 배치 평가, board copy 비용 등)는 나중에 프로파일링 후 결정
(docs/IDEAS.md 참고) — 지금은 시뮬레이션마다 leaf 하나씩 순차 평가하는 단순한 버전.
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
    exploration = C_PUCT * child.prior * math.sqrt(parent_visit_count) / (1 + child.visit_count)
    return child.value + exploration


def _select_child(node: Node) -> tuple[chess.Move, Node]:
    """Selection 단계: 자식들 중 PUCT 점수가 가장 높은 (수, 자식 노드)를 고른다."""
    parent_visit_count = sum(child.visit_count for child in node.children.values())
    return max(node.children.items(), key=lambda item: _puct_score(parent_visit_count, item[1]))


@torch.no_grad()
def _evaluate(board: chess.Board, model, device: str) -> tuple[dict[chess.Move, float], float]:
    """Expansion + Evaluation 단계: 리프 국면을 network로 평가.

    반환값은 (합법수별 prior, board.turn 관점 value) — 두 값 모두 "지금 이 국면에서
    둘 차례인 쪽"의 관점이다(engine 전반의 관례와 동일, 예: OnlineValuePolicy).
    """
    planes = encode_board(board)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    policy_logits, value = model(x)

    legal_moves = list(board.legal_moves)
    move_logits = np.array([policy_logits[0, MOVE_TO_INDEX[move.uci()]].item() for move in legal_moves])
    probs = np.exp(move_logits - move_logits.max())
    probs /= probs.sum()

    priors = {move: float(prob) for move, prob in zip(legal_moves, probs)}
    return priors, value.item()


def _terminal_value(board: chess.Board) -> float:
    """게임이 끝난 국면의 값을, board.turn(다음에 둘 차례인 쪽) 관점으로 반환."""
    result = board.result()
    if result == "1/2-1/2":
        return 0.0
    white_won = result == "1-0"
    return 1.0 if white_won == (board.turn == chess.WHITE) else -1.0


def run(board: chess.Board, model, num_simulations: int, device: str = "cpu") -> dict:
    """루트 board에서 num_simulations번 MCTS 탐색 후 방문분포와 root value를 반환.

    반환: {"visit_counts": {move_uci: N(root, a)}, "root_value": root 국면에 대한
    network의 board.turn 관점 value 추정치(탐색 이전, 참고용)}.
    """
    root = Node()
    root_priors, root_value = _evaluate(board, model, device)
    for move, prior in root_priors.items():
        root.children[move] = Node(prior=prior)

    for _ in range(num_simulations):
        sim_board = board.copy(stack=False)
        node = root
        path = []  # root -> leaf로 실제로 선택된 (edge) 노드들. root 자신은 edge가 아니므로 제외.

        while node.children:
            move, node = _select_child(node)
            sim_board.push(move)
            path.append(node)

        if sim_board.is_game_over():
            value = _terminal_value(sim_board)
        else:
            leaf_priors, value = _evaluate(sim_board, model, device)
            for move, prior in leaf_priors.items():
                node.children[move] = Node(prior=prior)

        # Backup: leaf에서 root 방향으로 값을 되돌린다. value는 leaf 국면에서
        # "다음에 둘 차례인 쪽"(= leaf로 들어온 수를 둔 사람의 상대) 관점이므로,
        # leaf 자신의 edge 통계(둔 사람 관점)에는 부호를 뒤집어 반영하고, 한 단계
        # 올라갈 때마다 관점이 다시 바뀌므로 부호도 다시 뒤집는다.
        sign = -1
        for visited in reversed(path):
            visited.visit_count += 1
            visited.value_sum += sign * value
            sign *= -1

    visit_counts = {move.uci(): child.visit_count for move, child in root.children.items()}
    return {"visit_counts": visit_counts, "root_value": root_value}
