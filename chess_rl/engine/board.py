"""python-chess board를 감싸서 RL 학습에 필요한 encode/decode를 제공하는 얇은 wrapper."""

from collections import OrderedDict

import chess
import chess.polyglot
import numpy as np

NUM_PIECE_PLANES = 12  # 6 piece types x 2 colors
NUM_INPUT_PLANES = 13  # 기물 12 + 차례(side-to-move) 1

# legal_moves_and_mask()의 zobrist hash 기준 LRU 캐시. MCTS가 transposition table 없이
# 트리를 뻗어나가도(chess_rl/mcts/node.py) 서로 다른 트리 경로/서로 다른 게임이 같은
# 포지션(특히 오프닝)에 반복해서 도달하는 일이 흔해서, 이 캐시가 실전에서 잘 맞는다.
# 여러 스레드(viz 서버의 실시간 대국 + 백그라운드 비교 워커)가 동시에 건드릴 수 있지만
# lock은 안 건다 — 최악의 경우 레이스로 캐시가 한 번 덜 맞는 것뿐, 정확성 문제는 없다
# (같은 board는 항상 같은 legal moves를 내므로 값 자체는 어느 스레드가 계산해도 동일).
#
# maxsize는 메모리 예산으로 정한다 — mask 하나가 ACTION_SPACE_SIZE(20,160)짜리 float32라
# 그것만으로 엔트리당 ~80KB(실측 .tmp_scripts/measure_cache_memory.py 기준 ~64KB/entry,
# 리스트 오버헤드 포함). 처음엔 200,000으로 뒀다가 실제로 10GB+ 먹어서 OOM으로 프로세스가
# 죽는 걸 겪었다 — 5,000이면 최악의 경우에도 ~300MB 수준.
_LEGAL_MOVES_CACHE_MAXSIZE = 5_000
_legal_moves_cache: "OrderedDict[int, tuple[list, np.ndarray]]" = OrderedDict()


def encode_board(board: chess.Board) -> np.ndarray:
    """board를 (13, 8, 8) plane으로 변환 — 기물 12(one-hot) + 차례 1.

    마지막 plane(index 12)은 백 차례면 전부 1, 흑 차례면 전부 0. value head의 학습
    target이 "둘 차례인 쪽 관점"이라서 차례 정보가 입력에 없으면 같은 기물 배치가
    차례에 따라 정반대 target을 받는 모순이 생긴다(초기 12-plane 인코딩의 결함,
    AlphaZero 원 논문도 side-to-move plane을 포함).
    """
    planes = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = chess.square_rank(square), chess.square_file(square)
        plane_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        planes[plane_idx, rank, file] = 1.0
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0
    return planes


def legal_move_mask(board: chess.Board, move_to_index: dict) -> np.ndarray:
    """현재 국면의 합법수에 대응하는 index만 1인 mask를 반환."""
    mask = np.zeros(len(move_to_index), dtype=np.float32)
    for move in board.legal_moves:
        mask[move_to_index[move.uci()]] = 1.0
    return mask


def legal_moves_and_mask(
    board: chess.Board, move_to_index: dict
) -> tuple[list, np.ndarray]:
    """(합법수 리스트, action space 크기의 float32 마스크)를 반환.

    같은 포지션(zobrist hash 기준 — 캐슬링권/앙파상/차례까지 포함하지만 수순 기록이나
    halfmove clock 등 legal move와 무관한 값은 무시)이 다시 나오면 python-chess로
    다시 계산하지 않고 캐시에서 바로 꺼낸다. 반환된 mask는 호출한 쪽에서 수정하지 말 것
    (캐시에 저장된 것과 같은 배열).
    """
    key = chess.polyglot.zobrist_hash(board)
    cached = _legal_moves_cache.get(key)
    if cached is not None:
        _legal_moves_cache.move_to_end(key)
        return cached

    legal_moves = list(board.legal_moves)
    mask = np.zeros(len(move_to_index), dtype=np.float32)
    for move in legal_moves:
        mask[move_to_index[move.uci()]] = 1.0

    _legal_moves_cache[key] = (legal_moves, mask)
    if len(_legal_moves_cache) > _LEGAL_MOVES_CACHE_MAXSIZE:
        _legal_moves_cache.popitem(last=False)
    return legal_moves, mask


def terminal_value_for_side_to_move(board: chess.Board) -> float:
    """종료된 국면(board.is_game_over() 전제)의 실제 결과를, board.turn(다음에 둘
    차례인 쪽 — 게임이 끝났으니 실제로 두진 못하지만) 관점의 값으로 반환.

    network은 체크메이트 같은 종료 국면을 특별히 학습한 적이 없어서 raw forward
    pass로는 신뢰할 수 없다 — 종료 국면의 값은 항상 이 함수로 직접 계산해야 한다
    (network에 다시 물어보지 말 것). MCTS(chess_rl/mcts/search.py)와
    OnlineValuePolicy의 화면 표시용 평가(value_estimate_white_perspective,
    move_values) 양쪽에서 공유한다.
    """
    result = board.result()
    if result == "1/2-1/2":
        return 0.0
    white_won = result == "1-0"
    return 1.0 if white_won == (board.turn == chess.WHITE) else -1.0
