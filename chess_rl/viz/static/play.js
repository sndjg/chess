let state = null;
let sessionId = null;
let selectedSquare = null;
let pendingPromotionMove = null;
let valueHistory = []; // [{ply, value}], value는 백 관점

function isPromotionMove(fen, fromSquare, toSquare) {
  const grid = parseFen(fen);
  const file = fromSquare.charCodeAt(0) - "a".charCodeAt(0);
  const row = 8 - Number(fromSquare[1]);
  const piece = grid[row][file];
  const toRank = Number(toSquare[1]);
  return (piece === "P" && toRank === 8) || (piece === "p" && toRank === 1);
}

function showPromotionPicker() {
  document.getElementById("promotion-picker").style.display = "flex";
}

function hidePromotionPicker() {
  document.getElementById("promotion-picker").style.display = "none";
}

function renderValueChart(svgEl) {
  svgEl.innerHTML = "";
  if (valueHistory.length === 0) return;

  const width = 320;
  const height = 80;
  const maxPly = Math.max(valueHistory[valueHistory.length - 1].ply, 1);

  const toXY = (point) => {
    const x = (point.ply / maxPly) * width;
    const y = height / 2 - (point.value / 1.0) * (height / 2 - 4);
    return [x, y];
  };

  const zeroLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
  zeroLine.setAttribute("x1", 0);
  zeroLine.setAttribute("y1", height / 2);
  zeroLine.setAttribute("x2", width);
  zeroLine.setAttribute("y2", height / 2);
  zeroLine.setAttribute("stroke", "#ccc");
  svgEl.appendChild(zeroLine);

  const points = valueHistory.map((p) => toXY(p).join(",")).join(" ");
  const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  polyline.setAttribute("points", points);
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "#2b7de9");
  polyline.setAttribute("stroke-width", 2);
  svgEl.appendChild(polyline);
}

function render() {
  const flipped = state.human_color === "black";

  const boardEl = document.getElementById("board");
  renderBoard(boardEl, state.fen, { selectedSquare, onSquareClick, flipped });

  const colorLabel = state.human_color === "white" ? "백" : "흑";
  document.getElementById("status-label").textContent = state.game_over
    ? `게임 종료: ${state.result} (당신: ${colorLabel})`
    : state.turn === state.human_color
      ? `당신 차례 (${colorLabel})`
      : `AI 차례 (당신: ${colorLabel})`;

  const thoughtBoardEl = document.getElementById("board-ai-thought");
  const arrowSvg = document.getElementById("arrow-overlay");
  if (state.fen_before_ai_move) {
    renderBoard(thoughtBoardEl, state.fen_before_ai_move, { flipped });
    renderMoveArrows(arrowSvg, state.ai_candidate_moves, { flipped });
    document.getElementById("ai-thought-label").textContent = state.ai_move
      ? `AI가 실제로 둔 수: ${state.ai_move}`
      : "";
  } else {
    renderBoard(thoughtBoardEl, state.fen, { flipped });
    arrowSvg.innerHTML = "";
    document.getElementById("ai-thought-label").textContent = "(아직 AI가 생각하지 않음)";
  }

  const learningPanel = document.getElementById("learning-panel");
  if (state.value_estimate !== undefined && state.value_estimate !== null) {
    learningPanel.style.display = "flex";
    document.getElementById("value-gauge").textContent = state.value_estimate.toFixed(3);
  } else {
    learningPanel.style.display = "none";
  }
  if (state.games_trained !== undefined && state.games_trained !== null) {
    document.getElementById("games-trained").textContent = state.games_trained;
  }
  if (state.training) {
    document.getElementById("loss-before").textContent = state.training.loss_before.toFixed(4);
    document.getElementById("loss-after").textContent = state.training.loss_after.toFixed(4);
  }

  renderValueChart(document.getElementById("value-chart"));
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

  if (isPromotionMove(state.fen, move.slice(0, 2), move.slice(2, 4))) {
    pendingPromotionMove = move;
    render();
    showPromotionPicker();
    return;
  }
  submitMove(move);
}

function pushHistoryPoint(value) {
  if (value === undefined || value === null) return;
  const ply = valueHistory.length > 0 ? valueHistory[valueHistory.length - 1].ply + 1 : 0;
  valueHistory.push({ ply, value });
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

  pushHistoryPoint(data.value_after_human_move);
  pushHistoryPoint(data.value_after_ai_move);
  state = { ...data, value_estimate: data.value_after_ai_move ?? data.value_after_human_move };
  render();
}

async function newGame() {
  const colorChoice = document.getElementById("color-select").value;
  const color = colorChoice === "random" ? (Math.random() < 0.5 ? "white" : "black") : colorChoice;
  const policy = document.getElementById("policy-select").value;
  const res = await fetch("/api/play/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ human_color: color, policy }),
  });
  const data = await res.json();
  sessionId = data.session_id;
  valueHistory = [];
  pushHistoryPoint(data.value_after_ai_move);
  state = { ...data, value_estimate: data.value_after_ai_move };
  selectedSquare = null;
  pendingPromotionMove = null;
  hidePromotionPicker();
  document.getElementById("message").textContent = "";
  document.getElementById("loss-before").textContent = "-";
  document.getElementById("loss-after").textContent = "-";
  render();
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("new-game-btn").addEventListener("click", newGame);
  document.querySelectorAll("#promotion-picker button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const piece = btn.dataset.piece;
      hidePromotionPicker();
      const move = pendingPromotionMove;
      pendingPromotionMove = null;
      submitMove(move + piece);
    });
  });
  newGame();
});
