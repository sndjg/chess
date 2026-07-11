"""self-play 리플레이 viz 로컬 서버 실행."""

import argparse

import uvicorn

from chess_rl.viz.server import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-dir", default="games")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="0.0.0.0으로 바꾸면 Tailscale 등 다른 기기에서 접근 가능 (같은 네트워크의 다른 기기에도 노출되니 주의)",
    )
    args = parser.parse_args()

    uvicorn.run(create_app(args.games_dir), host=args.host, port=args.port)
