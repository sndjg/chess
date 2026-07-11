"""MCTS 탐색 트리 노드.

트리는 transposition table 없이 단순 트리로 구성한다(같은 포지션이 다른 경로로
나와도 병합하지 않음) — AlphaZero 원조와 동일한 단순화.

한 Node는 "부모 국면에서 자신으로 오는 수(edge)"의 통계를 들고 있다. 즉
node.prior/visit_count/value_sum은 그 수를 둔 쪽(parent의 차례였던 플레이어)
관점의 값이다.
"""

import chess


class Node:
    def __init__(self, prior: float = 0.0):
        self.prior = prior  # P(s, a): network가 이 수에 준 사전 확률
        self.visit_count = 0  # N(s, a)
        self.value_sum = 0.0  # W(s, a): 누적 value
        self.children: dict[chess.Move, "Node"] = {}

    @property
    def value(self) -> float:
        """Q(s, a) = W / N. 아직 방문하지 않았으면 0으로 취급."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
