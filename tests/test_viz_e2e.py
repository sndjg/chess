"""headless Chromium으로 viz 웹앱을 실제로 구동해 렌더링/상호작용을 검증하는 E2E 테스트."""

import socket
import threading
import time

import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from chess_rl.rollout.game_record import GameRecord
from chess_rl.viz.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def viz_server_url(tmp_path):
    record = GameRecord(moves=["e2e4", "e7e5", "g1f3", "b8c6"], result="*")
    record.to_json(tmp_path / "sample.json")

    port = _free_port()
    config = uvicorn.Config(create_app(games_dir=str(tmp_path)), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        yield page
        browser.close()


def test_board_renders_start_position(viz_server_url, page):
    page.goto(viz_server_url)
    page.wait_for_selector("#board .square")

    squares = page.query_selector_all("#board .square")
    assert len(squares) == 64

    piece_texts = [sq.inner_text() for sq in squares if sq.inner_text()]
    assert len(piece_texts) == 32  # 시작 국면 기물 32개
    assert page.inner_text("#ply-label") == "0 / 4"


def test_slider_and_next_button_advance_ply(viz_server_url, page):
    page.goto(viz_server_url)
    page.wait_for_selector("#board .square")

    page.click("#next-btn")
    assert page.inner_text("#ply-label") == "1 / 4"
    assert page.inner_text("#move-label") == "e2e4"

    page.evaluate(
        """() => {
            const slider = document.getElementById('slider');
            slider.value = 4;
            slider.dispatchEvent(new Event('input'));
        }"""
    )
    assert page.inner_text("#ply-label") == "4 / 4"
    assert page.inner_text("#move-label") == "b8c6"
