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

const SVG_NS = "http://www.w3.org/2000/svg";

function makeSvgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

// 차트 하나의 축(가로/세로 선 + 눈금 라벨)을 그린다. plot 영역은 라벨 자리를 위해
// 전체 SVG보다 좌/하에 여백을 둔 안쪽 사각형.
function drawAxes(svgEl, { plotLeft, plotTop, plotRight, plotBottom, xLabels, yLabels }) {
  svgEl.appendChild(
    makeSvgEl("line", {
      x1: plotLeft,
      y1: plotTop,
      x2: plotLeft,
      y2: plotBottom,
      stroke: "#888",
    })
  );
  svgEl.appendChild(
    makeSvgEl("line", {
      x1: plotLeft,
      y1: plotBottom,
      x2: plotRight,
      y2: plotBottom,
      stroke: "#888",
    })
  );

  for (const { value, y } of yLabels) {
    const text = makeSvgEl("text", {
      x: plotLeft - 4,
      y: y + 3,
      "text-anchor": "end",
      "font-size": 9,
      fill: "#666",
    });
    text.textContent = value;
    svgEl.appendChild(text);
  }
  for (const { value, x } of xLabels) {
    const text = makeSvgEl("text", {
      x,
      y: plotBottom + 11,
      "text-anchor": "middle",
      "font-size": 9,
      fill: "#666",
    });
    text.textContent = value;
    svgEl.appendChild(text);
  }
}

function renderValueChart(svgEl) {
  svgEl.innerHTML = "";
  if (valueHistory.length === 0) return;

  const width = 320;
  const height = 100;
  const plotLeft = 26;
  const plotRight = width - 4;
  const plotTop = 4;
  const plotBottom = height - 14;
  const maxPly = Math.max(valueHistory[valueHistory.length - 1].ply, 1);

  const toXY = (point) => {
    const x = plotLeft + (point.ply / maxPly) * (plotRight - plotLeft);
    const y = (plotTop + plotBottom) / 2 - (point.value / 1.0) * ((plotBottom - plotTop) / 2);
    return [x, y];
  };

  drawAxes(svgEl, {
    plotLeft,
    plotTop,
    plotRight,
    plotBottom,
    xLabels: [
      { value: "0", x: plotLeft },
      { value: `${maxPly}수`, x: plotRight },
    ],
    yLabels: [
      { value: "+1", y: plotTop },
      { value: "0", y: (plotTop + plotBottom) / 2 },
      { value: "-1", y: plotBottom },
    ],
  });

  const zeroLine = makeSvgEl("line", {
    x1: plotLeft,
    y1: (plotTop + plotBottom) / 2,
    x2: plotRight,
    y2: (plotTop + plotBottom) / 2,
    stroke: "#ddd",
  });
  svgEl.appendChild(zeroLine);

  const points = valueHistory.map((p) => toXY(p).join(",")).join(" ");
  const polyline = makeSvgEl("polyline", {
    points,
    fill: "none",
    stroke: "#2b7de9",
    "stroke-width": 2,
  });
  svgEl.appendChild(polyline);
}

function formatMoveList(movesSan) {
  if (!movesSan || movesSan.length === 0) return "(아직 둔 수 없음)";
  const parts = [];
  for (let i = 0; i < movesSan.length; i += 2) {
    const moveNumber = i / 2 + 1;
    const white = movesSan[i];
    const black = movesSan[i + 1];
    parts.push(black ? `${moveNumber}. ${white} ${black}` : `${moveNumber}. ${white}`);
  }
  return parts.join(" ");
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
    document.getElementById("ai-thought-label").textContent = state.ai_move_san
      ? `AI가 실제로 둔 수: ${state.ai_move_san}`
      : "";
  } else {
    renderBoard(thoughtBoardEl, state.fen, { flipped });
    arrowSvg.innerHTML = "";
    document.getElementById("ai-thought-label").textContent = "(아직 AI가 생각하지 않음)";
  }

  document.getElementById("move-list").textContent = formatMoveList(state.moves_san);

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

function renderComparisonChart(history) {
  const svgEl = document.getElementById("comparison-chart");
  svgEl.innerHTML = "";
  if (!history || history.length === 0) return;

  const width = 320;
  const height = 100;
  const plotLeft = 26;
  const plotRight = width - 4;
  const plotTop = 4;
  const plotBottom = height - 14;
  const maxGames = Math.max(...history.map((h) => h.opponent_games_trained), 1);
  const lastIndex = Math.max(history.length - 1, 1);

  const toXY = (point, index) => {
    const x = plotLeft + (index / lastIndex) * (plotRight - plotLeft);
    const best = point.best_beaten_games_trained || 0;
    const y = plotBottom - (best / maxGames) * (plotBottom - plotTop);
    return [x, y];
  };

  drawAxes(svgEl, {
    plotLeft,
    plotTop,
    plotRight,
    plotBottom,
    xLabels: [
      { value: "1회", x: plotLeft },
      { value: `${history.length}회`, x: plotRight },
    ],
    yLabels: [
      { value: `${maxGames}판`, y: plotTop },
      { value: "0판", y: plotBottom },
    ],
  });

  const points = history.map((p, i) => toXY(p, i).join(",")).join(" ");
  const polyline = makeSvgEl("polyline", {
    points,
    fill: "none",
    stroke: "#2b7de9",
    "stroke-width": 2,
  });
  svgEl.appendChild(polyline);

  // 이긴 매치는 파란 점, 진 매치는 빈 점으로 같이 표시.
  history.forEach((point, i) => {
    const [x, y] = toXY(point, i);
    svgEl.appendChild(
      makeSvgEl("circle", {
        cx: x,
        cy: y,
        r: 3,
        fill: point.won ? "#2b7de9" : "none",
        stroke: "#2b7de9",
      })
    );
  });
}

function renderComparison(cmp) {
  const panel = document.getElementById("comparison-panel");
  const body = document.getElementById("comparison-body");

  if (cmp.status === "idle") {
    panel.style.display = "none";
    return;
  }
  panel.style.display = "flex";

  if (cmp.status === "no_opponent") {
    body.textContent = "비교할 다른 계보(family)가 아직 없습니다.";
    renderComparisonChart([]);
    return;
  }
  if (cmp.status === "error") {
    body.textContent = `비교 실패: ${cmp.error}`;
    renderComparisonChart(cmp.history || []);
    return;
  }

  const statusLabel = cmp.status === "running" ? "측정 중..." : "최근 측정 완료";
  const best =
    cmp.best_beaten_games_trained !== null && cmp.best_beaten_games_trained !== undefined
      ? `${cmp.best_beaten_games_trained}판째까지 이김`
      : "아직 이긴 상대 없음";
  body.textContent =
    `${statusLabel} — ${cmp.own_family}(${cmp.own_games_trained}판) vs ${cmp.opponent_family} — ` +
    `지금까지: ${best} (매치 ${cmp.history.length}회 진행)`;

  renderComparisonChart(cmp.history);
}

async function pollComparison() {
  try {
    const res = await fetch("/api/comparison");
    if (!res.ok) return;
    renderComparison(await res.json());
  } catch {
    // 폴링 실패는 조용히 무시하고 다음 주기에 재시도
  }
}

const LOG_LEVEL_SEVERITY = { debug: 10, info: 20, warning: 30, error: 40 };
let latestLogEntries = [];

function renderLogs() {
  const logBody = document.getElementById("log-body");
  const minSeverity =
    LOG_LEVEL_SEVERITY[document.getElementById("log-level-select").value];
  const wasScrolledToBottom =
    logBody.scrollHeight - logBody.scrollTop - logBody.clientHeight < 20;

  logBody.textContent = latestLogEntries
    .filter((entry) => LOG_LEVEL_SEVERITY[entry.level] >= minSeverity)
    .map((entry) => `[${entry.timestamp}] ${entry.level.toUpperCase()} ${entry.message}`)
    .join("\n");

  if (wasScrolledToBottom) {
    logBody.scrollTop = logBody.scrollHeight;
  }
}

async function pollLogs() {
  try {
    const res = await fetch("/api/logs");
    if (!res.ok) return;
    const data = await res.json();
    latestLogEntries = data.lines;
    renderLogs();
  } catch {
    // 폴링 실패는 조용히 무시하고 다음 주기에 재시도
  }
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
  pollComparison();
  setInterval(pollComparison, 5000);
  document.getElementById("log-level-select").addEventListener("change", renderLogs);
  pollLogs();
  setInterval(pollLogs, 3000);
});
