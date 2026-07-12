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
  // 판별 학습 loss는 학습이 백그라운드로 옮겨져 응답에 안 실림 — 서버 로그 패널로 확인.

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
  // 1단계: 사람 수만 서버에 적용하고 즉시 렌더링 — AI 탐색(수 초)을 기다리는 동안
  // 사람 수가 보드에 안 보이는 문제를 피한다. AI 수는 2단계에서 준비되는 대로 렌더링.
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
  // 직전 AI 생각 스냅샷(fen_before_ai_move 등)은 유지한 채 사람 수만 갱신.
  state = { ...state, ...data, value_estimate: data.value_after_human_move };
  render();

  if (data.game_over || data.turn === data.human_color) return;

  // 2단계: AI 응수 요청. 그 사이 상태 라벨은 "AI 차례"로 표시돼 있음.
  const aiRes = await fetch(`/api/play/${sessionId}/ai-move`, { method: "POST" });
  const aiData = await aiRes.json();
  if (!aiRes.ok) {
    document.getElementById("message").textContent = aiData.detail;
    return;
  }
  pushHistoryPoint(aiData.value_after_ai_move);
  state = { ...state, ...aiData, value_estimate: aiData.value_after_ai_move };
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
  render();
}

// 비교 워커가 우선순위 큐(최신 checkpoint 우선)로 처리하다 보니, 매치가 끝나서
// history에 쌓이는 순서가 실제 학습 진행 순서(own_games_trained 오름차순)와 다를 수 있다
// (예: 2, 5, 4, 3판째 순으로 처리됨). 그래프는 항상 실제 진행 순서를 보여줘야 하므로,
// own_games_trained 기준으로 다시 정렬하고 "지금까지 이긴 것" 누적값도 그 순서대로
// 새로 계산한다(백엔드가 처리 순서대로 저장해둔 값은 그대로 못 씀).
function chronologicalHistory(history) {
  const sorted = [...history].sort((a, b) => a.own_games_trained - b.own_games_trained);
  let best = null;
  let bestEpochs = null;
  return sorted.map((point) => {
    if (point.won && (best === null || point.opponent_games_trained > best)) {
      best = point.opponent_games_trained;
      bestEpochs = point.opponent_total_epochs ?? null;
    }
    return {
      ...point,
      best_beaten_games_trained: best,
      best_beaten_total_epochs: bestEpochs,
    };
  });
}

function renderComparisonChart(history) {
  const svgEl = document.getElementById("comparison-chart");
  svgEl.innerHTML = "";
  if (!history || history.length === 0) return;

  const useEpochs = historyUsesEpochs(history);
  const bestOf = useEpochs
    ? (h) => h.best_beaten_total_epochs || 0
    : (h) => h.best_beaten_games_trained || 0;
  const unit = useEpochs ? "ep" : "판";

  const width = 320;
  const height = 100;
  const plotLeft = 26;
  const plotRight = width - 4;
  const plotTop = 4;
  const plotBottom = height - 14;
  const maxAmount = Math.max(...history.map(opponentAmount), 1);
  const lastIndex = Math.max(history.length - 1, 1);

  const toXY = (point, index) => {
    const x = plotLeft + (index / lastIndex) * (plotRight - plotLeft);
    const y = plotBottom - (bestOf(point) / maxAmount) * (plotBottom - plotTop);
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
      { value: `${formatAmount(maxAmount)}${unit}`, y: plotTop },
      { value: `0${unit}`, y: plotBottom },
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

// 학습량 축: 누적 학습 epoch이 있으면 그걸 쓰고(판마다 조기 중단으로 학습량이 달라서
// epoch이 더 정확), 예전 checkpoint처럼 epoch 정보가 없으면 판 수로 fallback.
function ownAmount(point) {
  return point.own_total_epochs ?? point.own_games_trained;
}
function opponentAmount(point) {
  return point.opponent_total_epochs ?? point.opponent_games_trained;
}
function historyUsesEpochs(history) {
  return history.every(
    (h) =>
      h.own_total_epochs !== null &&
      h.own_total_epochs !== undefined &&
      h.opponent_total_epochs !== null &&
      h.opponent_total_epochs !== undefined
  );
}
function formatAmount(value) {
  return value >= 10000 ? `${Math.round(value / 1000)}k` : `${value}`;
}

// 매치 순서가 아니라 "우리 checkpoint가 학습량 k일 때 상대 checkpoint 학습량 m을
// 이겼는지"를 산점도로 보여준다 — x=우리 학습량, y=상대 학습량. 두 축이 같은 단위라
// plot 영역을 정사각형으로 맞추고 y=x 참조선을 그린다(대각선 위쪽에 점이 있으면
// 우리가 더 적게 학습하고도 상대의 더 많이 학습된 checkpoint를 이겼다는 뜻).
function renderFrontierChart(history) {
  const svgEl = document.getElementById("frontier-chart");
  svgEl.innerHTML = "";
  if (!history || history.length === 0) return;

  const useEpochs = historyUsesEpochs(history);
  const xOf = useEpochs ? (h) => h.own_total_epochs : (h) => h.own_games_trained;
  const yOf = useEpochs
    ? (h) => h.opponent_total_epochs
    : (h) => h.opponent_games_trained;

  const plotSize = 120;
  const plotLeft = 30;
  const plotTop = 6;
  const plotRight = plotLeft + plotSize;
  const plotBottom = plotTop + plotSize;
  const maxVal = Math.max(...history.map((h) => Math.max(xOf(h), yOf(h))), 1);

  const toXY = (point) => {
    const x = plotLeft + (xOf(point) / maxVal) * plotSize;
    const y = plotBottom - (yOf(point) / maxVal) * plotSize;
    return [x, y];
  };

  drawAxes(svgEl, {
    plotLeft,
    plotTop,
    plotRight,
    plotBottom,
    xLabels: [
      { value: "0", x: plotLeft },
      { value: formatAmount(maxVal), x: plotRight },
    ],
    yLabels: [
      { value: formatAmount(maxVal), y: plotTop },
      { value: "0", y: plotBottom },
    ],
  });

  svgEl.appendChild(
    makeSvgEl("line", {
      x1: plotLeft,
      y1: plotBottom,
      x2: plotRight,
      y2: plotTop,
      stroke: "#bbb",
      "stroke-dasharray": "4,3",
    })
  );

  history.forEach((point) => {
    const [x, y] = toXY(point);
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
    renderFrontierChart([]);
    return;
  }
  if (cmp.status === "error") {
    const ordered = chronologicalHistory(cmp.history || []);
    body.textContent = `비교 실패: ${cmp.error}`;
    renderComparisonChart(ordered);
    renderFrontierChart(ordered);
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

  const ordered = chronologicalHistory(cmp.history);
  renderComparisonChart(ordered);
  renderFrontierChart(ordered);
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
    // 학습 완료(백그라운드, 판 종료 후 ~분 단위)를 다음 인터랙션 없이도 반영.
    if (data.games_trained !== undefined && data.games_trained !== null) {
      document.getElementById("games-trained").textContent = data.games_trained;
    }
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
