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

let state = { fens: [], moves: [], idx: 0 };

function render() {
  const grid = parseFen(state.fens[state.idx]);
  const boardEl = document.getElementById("board");
  boardEl.innerHTML = "";
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const sq = document.createElement("div");
      sq.className = "square " + ((r + c) % 2 === 0 ? "light" : "dark");
      const piece = grid[r][c];
      if (piece) sq.textContent = PIECE_UNICODE[piece];
      boardEl.appendChild(sq);
    }
  }
  document.getElementById("ply-label").textContent = `${state.idx} / ${state.fens.length - 1}`;
  document.getElementById("move-label").textContent =
    state.idx > 0 ? state.moves[state.idx - 1] : "(시작 국면)";
}

async function loadGame(gameId) {
  const res = await fetch(`/api/games/${gameId}`);
  const data = await res.json();
  state = { fens: data.fens, moves: data.moves, idx: 0 };
  const slider = document.getElementById("slider");
  slider.max = state.fens.length - 1;
  slider.value = 0;
  document.getElementById("result-label").textContent = data.result;
  render();
}

async function loadGameList() {
  const res = await fetch("/api/games");
  const games = await res.json();
  const select = document.getElementById("game-select");
  select.innerHTML = "";
  for (const g of games) {
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    select.appendChild(opt);
  }
  if (games.length > 0) loadGame(games[0]);
}

document.addEventListener("DOMContentLoaded", () => {
  loadGameList();
  document.getElementById("game-select").addEventListener("change", (e) => loadGame(e.target.value));
  document.getElementById("slider").addEventListener("input", (e) => {
    state.idx = Number(e.target.value);
    render();
  });
  document.getElementById("prev-btn").addEventListener("click", () => {
    state.idx = Math.max(0, state.idx - 1);
    document.getElementById("slider").value = state.idx;
    render();
  });
  document.getElementById("next-btn").addEventListener("click", () => {
    state.idx = Math.min(state.fens.length - 1, state.idx + 1);
    document.getElementById("slider").value = state.idx;
    render();
  });
});
