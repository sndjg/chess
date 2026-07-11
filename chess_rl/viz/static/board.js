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

function renderBoard(boardEl, fen, { selectedSquare = null, onSquareClick = null } = {}) {
  const grid = parseFen(fen);
  boardEl.innerHTML = "";
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const square = rowColToSquare(r, c);
      const sq = document.createElement("div");
      sq.className = "square " + ((r + c) % 2 === 0 ? "light" : "dark");
      if (square === selectedSquare) sq.classList.add("selected");
      sq.dataset.square = square;
      const piece = grid[r][c];
      if (piece) sq.textContent = PIECE_UNICODE[piece];
      if (onSquareClick) sq.addEventListener("click", () => onSquareClick(square));
      boardEl.appendChild(sq);
    }
  }
}
