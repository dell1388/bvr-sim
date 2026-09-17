"""Serve the HTTP API: python -m skysim [--host H] [--port P]"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="skysim", description="run the skysim API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("skysim.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
