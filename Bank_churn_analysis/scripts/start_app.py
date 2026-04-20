"""
scripts/start_app.py
--------------------
Convenience launcher.  Reads host/port from config/env and
starts uvicorn with appropriate settings for dev vs prod.

Usage:
    python scripts/start_app.py
    python scripts/start_app.py --prod
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import settings
from src.utils.logger import get_logger

log = get_logger("start_app")
S = settings


def start(prod: bool = False):
    host    = S.app.host
    port    = int(S.app.port)
    workers = 1 if not prod else 4
    reload  = not prod

    log.info(f"Starting FastAPI app on http://{host}:{port}")
    log.info(f"Mode: {'production' if prod else 'development'}")
    log.info(f"UI:   http://localhost:{port}/ui")
    log.info(f"Docs: http://localhost:{port}/docs")

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host",    host,
        "--port",    str(port),
        "--workers", str(workers),
    ]
    if reload:
        cmd.append("--reload")

    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prod", action="store_true",
                        help="Run in production mode (no reload, 4 workers)")
    args = parser.parse_args()
    start(prod=args.prod)
