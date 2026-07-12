const PIECE_UNICODE = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

function parseFen(fen) {
  const boardPart = fen.split(" ")[0];
  const rows = boardPart.split("/");
  const grid = [];
  for (const row of rows) {
    const line = [];
    for (const ch of row) {
      if (/[1-8]/.test(ch)) {
        for (let i = 0; i < Number(ch); i++) line.push(null);
      } else {
        line.push(ch);
      }
    }
    grid.push(line);
  }
  return grid; // grid[0] = 8랭크 ... grid[7] = 1랭크
}

function rowColToSquare(row, col) {
  const file = String.fromCharCode("a".charCodeAt(0) + col);
  const rank = 8 - row;
  return `${file}${rank}`;
}

const SQUARE_SIZE = 48;

function squareCenter(square, flipped = false) {
  const file = square.charCodeAt(0) - "a".charCodeAt(0);
  const rank = Number(square[1]);
  let row = 8 - rank;
  let col = file;
  if (flipped) {
    row = 7 - row;
    col = 7 - col;
  }
  return [col * SQUARE_SIZE + SQUARE_SIZE / 2, row * SQUARE_SIZE + SQUARE_SIZE / 2];
}

/** candidateMoves: [{move: "e2e4", value: 0.3}, ...] (내림차순 정렬 가정). 상위 topN개만 화살표로 그림. */
function renderMoveArrows(svgEl, candidateMoves, { topN = 8, flipped = false } = {}) {
  svgEl.innerHTML = "";
  if (!candidateMoves || candidateMoves.length === 0) return;

  const shown = candidateMoves.slice(0, topN);
  // MCTS 후보(visits 있음)면 방문 횟수로 진하기를 정한다 — 수 선택도 방문 횟수
  // argmax라서, 가장 진한 화살표 = 실제로 두는 수가 되도록. visits가 없는 후보
  // (value만 있는 예전 형식)는 value 기준 유지.
  const weightOf = (c) => (c.visits !== undefined ? c.visits : c.value);
  const weights = shown.map(weightOf);
  const min = Math.min(...weights);
  const max = Math.max(...weights);
  const span = max - min || 1;

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  svgEl.appendChild(defs);

  for (const candidate of shown) {
    const from = candidate.move.slice(0, 2);
    const to = candidate.move.slice(2, 4);
    const [x1, y1] = squareCenter(from, flipped);
    const [x2, y2] = squareCenter(to, flipped);
    const opacity = 0.2 + 0.8 * ((weightOf(candidate) - min) / span);

    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("stroke", "#2b7de9");
    line.setAttribute("stroke-width", 5);
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("opacity", opacity.toFixed(2));
    line.setAttribute("marker-end", "url(#arrowhead)");
    svgEl.appendChild(line);
  }

  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "arrowhead");
  marker.setAttribute("markerWidth", "6");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("refX", "4");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");
  const arrowPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
  arrowPath.setAttribute("d", "M0,0 L6,3 L0,6 Z");
  arrowPath.setAttribute("fill", "#2b7de9");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
}

function renderBoard(boardEl, fen, { selectedSquare = null, onSquareClick = null, flipped = false } = {}) {
  const grid = parseFen(fen);
  boardEl.innerHTML = "";
  for (let displayRow = 0; displayRow < 8; displayRow++) {
    for (let displayCol = 0; displayCol < 8; displayCol++) {
      const row = flipped ? 7 - displayRow : displayRow;
      const col = flipped ? 7 - displayCol : displayCol;
      const square = rowColToSquare(row, col);
      const sq = document.createElement("div");
      sq.className = "square " + ((row + col) % 2 === 0 ? "light" : "dark");
      if (square === selectedSquare) sq.classList.add("selected");
      sq.dataset.square = square;
      const piece = grid[row][col];
      if (piece) sq.textContent = PIECE_UNICODE[piece];
      if (onSquareClick) sq.addEventListener("click", () => onSquareClick(square));
      boardEl.appendChild(sq);
    }
  }
}
