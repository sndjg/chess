let state = { fens: [], moves: [], idx: 0 };

function render() {
  const boardEl = document.getElementById("board");
  renderBoard(boardEl, state.fens[state.idx]);

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
