import torch

from chess_rl.engine.action_space import ACTION_SPACE_SIZE
from chess_rl.model.network import PolicyValueNet
from chess_rl.utils.checkpoint import list_checkpoints, load_checkpoint, save_checkpoint


def _small_model():
    return PolicyValueNet(
        in_planes=12, action_space_size=ACTION_SPACE_SIZE, channels=8, num_blocks=1
    )


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = _small_model()
    path = save_checkpoint(model, str(tmp_path), games_trained=7)

    assert path.name == "game_000007.pt"
    loaded = load_checkpoint(path, device="cpu")

    x = torch.zeros(1, 12, 8, 8)
    with torch.no_grad():
        expected = model(x)
        actual = loaded(x)
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])


def test_list_checkpoints_sorted_ascending(tmp_path):
    model = _small_model()
    save_checkpoint(model, str(tmp_path), games_trained=20)
    save_checkpoint(model, str(tmp_path), games_trained=5)
    save_checkpoint(model, str(tmp_path), games_trained=10)

    checkpoints = list_checkpoints(str(tmp_path))
    assert [c.games_trained for c in checkpoints] == [5, 10, 20]


def test_list_checkpoints_empty_dir_returns_empty_list(tmp_path):
    assert list_checkpoints(str(tmp_path / "does_not_exist")) == []
