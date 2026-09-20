"""Kaggle launcher: start Voice Box exactly like production, but on 0.0.0.0.

This is the FastAPI server the Survée backend calls.  It uses the repository's
real entrypoint (``python -m backend.main``) so behavior matches local/Docker
deployment — the only difference is the bind host (0.0.0.0) and the explicit
--data-dir / HF_HUB_CACHE, which are invisible to application code.

Run from the Voice Box repository root:

    python kaggle/start_voicebox.py

Defaults (must match setup_voicebox.py):

    VOICEBOX_HOST         bind host             (default 0.0.0.0)
    VOICEBOX_PORT         port                  (default 17493)
    VOICEBOX_DATA_DIR     sqlite + audio dir    (default /kaggle/working/voicebox-data
                                                 on Kaggle, else ./data)
    VOICEBOX_MODELS_DIR   HF cache (HF_HUB_CACHE) (default /kaggle/working/voicebox-models
                                                 on Kaggle, else ./models-cache)

In a notebook, run this in its own cell as a background process so the server
stays alive while you verify:

    import subprocess
    subprocess.Popen(["python", "kaggle/start_voicebox.py"], stdout=open("/kaggle/working/voicebox.log", "w"))
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
IS_KAGGLE = os.path.isdir("/kaggle/working")

DATA_DIR = Path(
    os.environ.get("VOICEBOX_DATA_DIR")
    or ("/kaggle/working/voicebox-data" if IS_KAGGLE else str(Path("data").resolve()))
)
MODELS_DIR = Path(
    os.environ.get("VOICEBOX_MODELS_DIR")
    or ("/kaggle/working/voicebox-models" if IS_KAGGLE else str(Path("models-cache").resolve()))
)
HOST = os.environ.get("VOICEBOX_HOST", "0.0.0.0")
PORT = int(os.environ.get("VOICEBOX_PORT", "17493"))


def main() -> int:
    if not (REPO_ROOT / "backend").is_dir():
        print(f"ERROR: backend/ not found under {REPO_ROOT}. Run from the Voice Box repo root.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Must be set before backend.config is imported (it maps VOICEBOX_MODELS_DIR
    # to HF_HUB_CACHE at import time).
    os.environ.setdefault("VOICEBOX_MODELS_DIR", str(MODELS_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
    os.environ.setdefault("VOICEBOX_DATA_DIR", str(DATA_DIR))

    cmd = [
        sys.executable,
        "-m",
        "backend.main",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--data-dir",
        str(DATA_DIR),
    ]

    print(f"Starting Voice Box on http://{HOST}:{PORT} (data={DATA_DIR}, models={MODELS_DIR})")
    print("Ctrl+C to stop.")
    print(f"$ {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())