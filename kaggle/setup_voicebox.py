"""Kaggle bootstrap: install Voice Box deps, seed the Survée Vivian profile, pre-download the model.

Runs on a Kaggle GPU notebook or any Linux box with internet.

Run from the Voice Box repository root:

    python kaggle/setup_voicebox.py
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
    or (
        "/kaggle/working/voicebox-data"
        if IS_KAGGLE
        else str(Path("data").resolve())
    )
)

MODELS_DIR = Path(
    os.environ.get("VOICEBOX_MODELS_DIR")
    or (
        "/kaggle/working/voicebox-models"
        if IS_KAGGLE
        else str(Path("models-cache").resolve())
    )
)

TORCH_INDEX = os.environ.get(
    "VOICEBOX_TORCH_INDEX",
    "https://download.pytorch.org/whl/cu128",
)

VIVIAN_PROFILE_ID = "d69609bd-1117-4a39-8d9c-1f57368b8835"

VIVIAN_NAME = "Survée Waiter 1"

VIVIAN_DESCRIPTION = (
    "The official Survée AI Waiter voice for restaurant customers. "
    "Warm, professional, natural, confident, clear, and conversational."
)

QWEN_CUSTOM_VOICE_1_7B = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)


def _run(cmd: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip"] + cmd,
        **kwargs,
    )


def _torch_has_cuda() -> bool:
    try:
        import torch

        print(
            f"PyTorch {torch.__version__} already installed — "
            f"CUDA available: {torch.cuda.is_available()}"
        )

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")

            for index in range(torch.cuda.device_count()):
                print(
                    f"GPU {index}: "
                    f"{torch.cuda.get_device_name(index)}"
                )

        return torch.cuda.is_available()

    except ImportError:
        print("PyTorch is not installed yet.")
        return False


def main() -> int:
    if not (REPO_ROOT / "backend").is_dir():
        print(
            f"ERROR: backend/ not found under {REPO_ROOT}. "
            "Run from the Voice Box repo root."
        )
        return 1

    print("Voice Box Kaggle setup")
    print(f"  repo root     : {REPO_ROOT}")
    print(f"  data dir      : {DATA_DIR}")
    print(f"  models dir    : {MODELS_DIR}")
    print(f"  torch index   : {TORCH_INDEX}")

    # 1. Upgrade pip.
    _run(["install", "--upgrade", "pip"])

    # 2. Install CUDA-enabled PyTorch if necessary.
    if not _torch_has_cuda():
        _run(
            [
                "install",
                "torch",
                "torchaudio",
                "--index-url",
                TORCH_INDEX,
            ]
        )

    # 3. Install the repository's canonical backend dependencies.
    _run(
        [
            "install",
            "-r",
            str(REPO_ROOT / "backend" / "requirements.txt"),
        ]
    )

    # 4. Install Chatterbox and TADA without dependency resolution.
    _run(
        [
            "install",
            "--no-deps",
            "chatterbox-tts",
        ]
    )

    _run(
        [
            "install",
            "--no-deps",
            "hume-tada",
        ]
    )

    # 5. Install Qwen3-TTS.
    _run(
        [
            "install",
            "git+https://github.com/QwenLM/Qwen3-TTS.git",
        ]
    )

    # 6. Configure the Hugging Face model cache.
    os.environ["VOICEBOX_MODELS_DIR"] = str(MODELS_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 7. Make the Voice Box repository importable.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # 8. Configure the Voice Box data directory.
    from backend import config

    config.set_data_dir(DATA_DIR)

    # 9. Initialize the database.
    #
    # IMPORTANT:
    # SessionLocal is created dynamically by init_db().
    # Therefore we access it through the database module AFTER init_db()
    # instead of importing SessionLocal by value before initialization.
    from backend import database
    from backend.database import VoiceProfile, init_db

    init_db()

    if database.SessionLocal is None:
        raise RuntimeError(
            "Database initialization did not create SessionLocal."
        )

    db = database.SessionLocal()

    try:
        existing = db.get(
            VoiceProfile,
            VIVIAN_PROFILE_ID,
        )

        if existing is None:
            db.add(
                VoiceProfile(
                    id=VIVIAN_PROFILE_ID,
                    name=VIVIAN_NAME,
                    description=VIVIAN_DESCRIPTION,
                    language="zh",
                    voice_type="preset",
                    preset_engine="qwen_custom_voice",
                    preset_voice_id="Vivian",
                    design_prompt=None,
                    default_engine="qwen_custom_voice",
                    avatar_path=None,
                    personality=VIVIAN_DESCRIPTION,
                )
            )

            db.commit()

            print(
                f"Seeded profile "
                f"{VIVIAN_PROFILE_ID} "
                f"({VIVIAN_NAME})"
            )

        else:
            print(
                f"Profile {VIVIAN_PROFILE_ID} already present - "
                "not overwriting "
                f"(preset_engine={existing.preset_engine}, "
                f"preset_voice_id={existing.preset_voice_id})"
            )

    finally:
        db.close()

    # 10. Pre-download Qwen CustomVoice 1.7B.
    from huggingface_hub import snapshot_download

    print(
        f"\nPre-downloading "
        f"{QWEN_CUSTOM_VOICE_1_7B} "
        f"into {MODELS_DIR} ..."
    )

    path = snapshot_download(
        repo_id=QWEN_CUSTOM_VOICE_1_7B,
        cache_dir=str(MODELS_DIR),
    )

    print(f"Model ready at: {path}")

    # 11. Verify Qwen3-TTS can be imported.
    from qwen_tts import Qwen3TTSModel  # noqa: F401

    print("\n========================================")
    print("Voice Box Kaggle setup COMPLETE")
    print("========================================")
    print()
    print("Start the server with:")
    print("  python kaggle/start_voicebox.py")
    print()
    print("Then verify with:")
    print("  python kaggle/verify_voicebox.py")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())