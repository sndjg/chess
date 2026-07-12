"""AlphaZero 스타일 policy+value network (작은 ResNet)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_planes: int,
        action_space_size: int,
        channels: int = 128,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, action_space_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value


# 표준 기물 가치. index = chess.PieceType - 1 (P, N, B, R, Q, K).
_PIECE_VALUES = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)


class MaterialBlendedPolicyValueNet(nn.Module):
    """value를 '신경망 value head'와 '남은 기물 점수 휴리스틱'의 가중합으로 내는 모델.

    v = (1 - material_weight) * v_nn + material_weight * v_material

    - v_material: 입력 plane에서 직접 계산(기물 plane별 합 x 표준 가치 -> 백 관점 재료
      차이 -> tanh(차이/scale)로 [-1,1] -> 차례 plane(index 12)으로 둘 차례인 쪽 관점으로
      부호 정렬). 별도 board 객체가 필요 없어 MCTS 배치 평가 경로에 그대로 들어간다.
    - 학습(MSE)은 가중합된 출력에 걸리므로 gradient는 신경망 쪽으로만 흐르고, 재료 항은
      고정 prior 역할 — 신경망은 재료 점수로 설명 안 되는 잔차를 학습하게 된다.
    - policy logits는 내부 PolicyValueNet 그대로 통과.

    입력은 13-plane 인코딩(engine.board.encode_board) 전제 — 차례 plane이 없으면 재료
    점수의 관점(부호)을 정할 수 없다.
    """

    def __init__(
        self,
        net: PolicyValueNet,
        material_weight: float = 0.5,
        material_scale: float = 10.0,
    ):
        super().__init__()
        if not 0.0 <= material_weight <= 1.0:
            raise ValueError(f"material_weight는 [0, 1]이어야 함: {material_weight}")
        self.net = net
        self.material_weight = material_weight
        self.material_scale = material_scale
        piece_values = torch.tensor(_PIECE_VALUES + _PIECE_VALUES, dtype=torch.float32)
        piece_values[6:] *= -1  # 흑 기물(plane 6~11)은 백 관점에서 음수
        self.register_buffer("_piece_values", piece_values)

    def material_value_for_side_to_move(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, 13, 8, 8) 입력에서 둘 차례인 쪽 관점의 재료 점수([-1,1])를 계산."""
        piece_counts = x[:, :12].sum(dim=(2, 3))  # (batch, 12)
        material_white = (piece_counts * self._piece_values).sum(dim=1)  # 백 관점
        white_to_move = x[:, 12, 0, 0]  # 차례 plane: 백 차례면 1, 흑 차례면 0
        sign = white_to_move * 2.0 - 1.0  # 1 -> +1(백 관점 그대로), 0 -> -1(부호 반전)
        return torch.tanh(material_white * sign / self.material_scale)

    def forward(self, x: torch.Tensor):
        policy_logits, nn_value = self.net(x)
        material_value = self.material_value_for_side_to_move(x)
        value = (
            1.0 - self.material_weight
        ) * nn_value + self.material_weight * material_value
        return policy_logits, value
