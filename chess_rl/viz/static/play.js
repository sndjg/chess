let state = null;
let sessionId = null;
let selectedSquare = null;

function render() {
  const boardEl = document.getElementById("board");
  renderBoard(boardEl, state.fen, { selectedSquare, onSquareClick });

  document.getElementById("status-label").textContent = state.game_over
    ? `게임 종료: ${state.result}`
    : state.turn === state.human_color
      ? "당신 차례"
      : "AI 차례";
}

function onSquareClick(square) {
  if (!state || state.game_over || state.turn !== state.human_color) return;

  if (selectedSquare === null) {
    selectedSquare = square;
    render();
    return;
  }
  if (selectedSquare === square) {
    selectedSquare = null;
    render();
    return;
  }

  const move = selectedSquare + square;
  selectedSquare = null;
  submitMove(move);
}

async function submitMove(uci) {
  const res = await fetch(`/api/play/${sessionId}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ move: uci }),
  });
  const data = await res.json();
  if (!res.ok) {
    document.getElementById("message").textContent = data.detail;
    return;
  }
  document.getElementById("message").textContent = "";
  state = data;
  render();
}

async function newGame() {
  const color = document.getElementById("color-select").value;
  const res = await fetch("/api/play/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ human_color: color, policy: "random" }),
  });
  const data = await res.json();
  sessionId = data.session_id;
  state = data;
  selectedSquare = null;
  document.getElementById("message").textContent = "";
  render();
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("new-game-btn").addEventListener("click", newGame);
  newGame();
});
