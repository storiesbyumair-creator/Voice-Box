"""Kaggle bootstrap: install Voice Box dependencies, seed the Survée Vivian
profile, and pre-download the Qwen CustomVoice model.

Run from the Voice Box repository root:

    python kaggle/setup_voicebox.py

Designed for Kaggle GPU notebooks, but also works on a Linux machine with
internet access.

Environment overrides:

    VOICEBOX_TORCH_INDEX
        PyTorch CUDA wheel index.
        Default: https://download.pytorch.org/whl/cu128

    VOICEBOX_DATA_DIR
        Voice Box SQLite/audio data directory.
        Kaggle default: /kaggle/working/voicebox-data
        Other default:  ./data

    VOICEBOX_MODELS_DIR
        Hugging Face model cache directory.
        Kaggle default: /kaggle/working/voicebox-models
        Other default:  ./models-cache
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
IS_KAGGLE = Path("/kaggle/working").is_dir()

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


# ---------------------------------------------------------------------------
# Survée Vivian profile
# ---------------------------------------------------------------------------

VIVIAN_PROFILE_ID = "d69609bd-1117-4a39-8d9c-1f57368b8835"

VIVIAN_NAME = "Survée Waiter 1"

VIVIAN_DESCRIPTION = (
    "The official Survée AI Waiter voice for restaurant customers. "
    "Warm, professional, natural, confident, clear, and conversational."
)

QWEN_CUSTOM_VOICE_MODEL = (
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pip(args: list[str], **kwargs) -> None:
    """Run pip using the same Python interpreter running this script."""
    command = [sys.executable, "-m", "pip", *args]

    print()
    print("$", " ".join(command))

    subprocess.check_call(command, **kwargs)


def _check_cuda() -> bool:
    """Check whether the currently installed PyTorch can access CUDA."""
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed.")
        return False

    cuda_available = torch.cuda.is_available()

    print()
    print("PyTorch environment")
    print("-------------------")
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", cuda_available)
    print("CUDA version    :", torch.version.cuda)

    if cuda_available:
        gpu_count = torch.cuda.device_count()

        print("GPU count       :", gpu_count)

        for index in range(gpu_count):
            print(
                f"GPU {index}         : "
                f"{torch.cuda.get_device_name(index)}"
            )

    return cuda_available


def _configure_paths() -> None:
    """Configure directories used by Voice Box and Hugging Face."""
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ["VOICEBOX_DATA_DIR"] = str(DATA_DIR)
    os.environ["VOICEBOX_MODELS_DIR"] = str(MODELS_DIR)

    os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)


def _seed_vivian_profile() -> None:
    """Initialize the Voice Box DB and seed the existing Vivian profile."""
    # Make the repository importable.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Configure Voice Box's data directory BEFORE importing/initializing
    # the database.
    from backend import config

    config.set_data_dir(DATA_DIR)

    # IMPORTANT:
    #
    # backend.database.__init__ re-exports SessionLocal:
    #
    #     from .session import SessionLocal
    #
    # That is a snapshot of the value at import time.
    #
    # backend.database.session.SessionLocal, however, is assigned dynamically
    # by init_db().
    #
    # Therefore we deliberately import the session MODULE and access
    # SessionLocal from that module AFTER init_db().
    from backend.database import VoiceProfile, init_db
    from backend.database import session as database_session

    print()
    print("Initializing Voice Box database...")
    init_db()

    if database_session.SessionLocal is None:
        raise RuntimeError(
            "Database initialization completed, but "
            "backend.database.session.SessionLocal is still None."
        )

    db = database_session.SessionLocal()

    try:
        existing = db.get(
            VoiceProfile,
            VIVIAN_PROFILE_ID,
        )

        if existing is None:
            profile = VoiceProfile(
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

            db.add(profile)
            db.commit()

            print()
            print("Vivian profile seeded successfully.")
            print("  ID     :", VIVIAN_PROFILE_ID)
            print("  Name   :", VIVIAN_NAME)
            print("  Engine :", "qwen_custom_voice")
            print("  Voice  :", "Vivian")

        else:
            print()
            print("Vivian profile already exists.")
            print("  ID     :", existing.id)
            print("  Name   :", existing.name)
            print("  Engine :", existing.preset_engine)
            print("  Voice  :", existing.preset_voice_id)
            print("  Action :", "existing profile preserved")

    finally:
        db.close()


def _download_qwen_model() -> None:
    """Download/cache the Qwen CustomVoice model."""
    from huggingface_hub import snapshot_download

    print()
    print("=" * 60)
    print("Downloading Qwen CustomVoice model")
    print("=" * 60)
    print("Model :", QWEN_CUSTOM_VOICE_MODEL)
    print("Cache :", MODELS_DIR)

    model_path = snapshot_download(
        repo_id=QWEN_CUSTOM_VOICE_MODEL,
        cache_dir=str(MODELS_DIR),
    )

    print()
    print("Qwen model is ready.")
    print("Path:", model_path)


def main() -> int:
    # -----------------------------------------------------------------------
    # Validate repository
    # -----------------------------------------------------------------------

    if not (REPO_ROOT / "backend").is_dir():
        print(
            f"ERROR: backend/ was not found under {REPO_ROOT}."
        )
        print(
            "Run this script from the Voice Box repository."
        )
        return 1

    print()
    print("=" * 60)
    print("VOICE BOX — KAGGLE GPU SETUP")
    print("=" * 60)
    print()
    print("Repository :", REPO_ROOT)
    print("Kaggle     :", IS_KAGGLE)
    print("Data dir   :", DATA_DIR)
    print("Models dir :", MODELS_DIR)
    print("Torch index:", TORCH_INDEX)

    # -----------------------------------------------------------------------
    # 1. pip
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("1. Updating pip")
    print("=" * 60)

    _run_pip(
        [
            "install",
            "--upgrade",
            "pip",
        ]
    )

    # -----------------------------------------------------------------------
    # 2. PyTorch / CUDA
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("2. Checking PyTorch / CUDA")
    print("=" * 60)

    cuda_ready = _check_cuda()

    if not cuda_ready:
        print()
        print("CUDA is not available.")
        print("Installing CUDA-enabled PyTorch...")

        _run_pip(
            [
                "install",
                "torch",
                "torchaudio",
                "--index-url",
                TORCH_INDEX,
            ]
        )

        print()
        print("Rechecking CUDA...")

        if not _check_cuda():
            raise RuntimeError(
                "PyTorch was installed, but CUDA is still unavailable."
            )

    # -----------------------------------------------------------------------
    # 3. Voice Box requirements
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("3. Installing Voice Box backend requirements")
    print("=" * 60)

    requirements_file = (
        REPO_ROOT
        / "backend"
        / "requirements.txt"
    )

    _run_pip(
        [
            "install",
            "-r",
            str(requirements_file),
        ]
    )

    # -----------------------------------------------------------------------
    # 4. Chatterbox / TADA
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("4. Installing Chatterbox / TADA pins")
    print("=" * 60)

    _run_pip(
        [
            "install",
            "--no-deps",
            "chatterbox-tts",
        ]
    )

    _run_pip(
        [
            "install",
            "--no-deps",
            "hume-tada",
        ]
    )

    # -----------------------------------------------------------------------
    # 5. Qwen3-TTS
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("5. Installing Qwen3-TTS")
    print("=" * 60)

    _run_pip(
        [
            "install",
            "git+https://github.com/QwenLM/Qwen3-TTS.git",
        ]
    )

    # -----------------------------------------------------------------------
    # 6. Configure cache/data paths
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("6. Configuring Voice Box directories")
    print("=" * 60)

    _configure_paths()

    print("VOICEBOX_DATA_DIR  =", DATA_DIR)
    print("VOICEBOX_MODELS_DIR =", MODELS_DIR)
    print("HF_HUB_CACHE       =", MODELS_DIR)

    # -----------------------------------------------------------------------
    # 7. Seed Vivian
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("7. Initializing database and Vivian profile")
    print("=" * 60)

    _seed_vivian_profile()

    # -----------------------------------------------------------------------
    # 8. Download Qwen model
    # -----------------------------------------------------------------------

    _download_qwen_model()

    # -----------------------------------------------------------------------
    # 9. Qwen import smoke test
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("9. Verifying Qwen3-TTS import")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel  # noqa: F401

    print("Qwen3TTSModel import: OK")

    # -----------------------------------------------------------------------
    # Complete
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("VOICE BOX KAGGLE SETUP COMPLETE")
    print("=" * 60)
    print()
    print("GPU is ready.")
    print("Vivian profile is ready.")
    print("Qwen CustomVoice model is cached.")
    print()
    print("Next:")
    print()
    print("  python kaggle/start_voicebox.py")
    print()
    print("Then:")
    print()
    print("  python kaggle/verify_voicebox.py")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())