"""Kaggle bootstrap: install Voice Box deps, seed the Survée Vivian profile, pre-download the model.

Runs on a Kaggle GPU notebook (or any Linux box with internet).  This script does
NOT recreate Voice Box — it drives the repository's own dependency definitions
(backend/requirements.txt, the PyTorch CUDA index used by the justfile, and the
QwenLM/Qwen3-TTS git install) and seeds one preset profile through the ORM.

Servable engine after setup: qwen_custom_voice / 1.7B => profiler "Vivian".

Run from the Voice Box repository root:

    python kaggle/setup_voicebox.py

Environment overrides (optional):

    VOICEBOX_TORCH_INDEX   PyTorch CUDA wheel index. Default:
                           https://download.pytorch.org/whl/cu128
                           (fall back to cu124/cu126 if torch cannot see CUDA)
    VOICEBOX_DATA_DIR      data dir for the server (sqlite + audio).
                           Default: /kaggle/working/voicebox-data (Kaggle),
                           ./data (elsewhere)
    VOICEBOX_MODELS_DIR    HuggingFace cache dir (HF_HUB_CACHE).
                           Default: /kaggle/working/voicebox-models (Kaggle),
                           ./models-cache (elsewhere)

The same defaults are used by start_voicebox.py, so setup and start always agree.
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
TORCH_INDEX = os.environ.get(
    "VOICEBOX_TORCH_INDEX", "https://download.pytorch.org/whl/cu128"
)

# Survée's active waiter voice — preset profile (exact UUID + values from the
# local dev DB). Preset profiles carry no reference audio; the row is all that
# is needed, plus the Qwen CustomVoice weights below.
VIVIAN_PROFILE_ID = "d69609bd-1117-4a39-8d9c-1f57368b8835"
VIVIAN_NAME = "Surv\u00e9e Waiter 1"
VIVIAN_DESCRIPTION = (
    "The official Surv\u00e9e AI Waiter voice for restaurant customers. "
    "Warm, professional, natural, confident, clear, and conversational."
)

QWEN_CUSTOM_VOICE_1_7B = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def _run(cmd: list[str], **kwargs) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call([sys.executable, "-m", "pip"] + cmd, **kwargs)


def _torch_has_cuda() -> bool:
    try:
        import torch

        print(
            f"PyTorch {torch.__version__} already installed — "
            f"CUDA available: {torch.cuda.is_available()}"
        )
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> int:
    if not (REPO_ROOT / "backend").is_dir():
        print(f"ERROR: backend/ not found under {REPO_ROOT}. Run from the Voice Box repo root.")
        return 1

    print("Voice Box Kaggle setup")
    print(f"  repo root     : {REPO_ROOT}")
    print(f"  data dir      : {DATA_DIR}")
    print(f"  models dir    : {MODELS_DIR}")
    print(f"  torch index   : {TORCH_INDEX}")

    # 1. pip itself.
    _run(["install", "--upgrade", "pip"])

    # 2. PyTorch with CUDA — mirrors the justfile Linux/NVIDIA branch
    #    (torch + torchaudio from the CUDA index).  Skip if already CUDA-ready.
    if not _torch_has_cuda():
        _run(["install", "torch", "torchaudio", "--index-url", TORCH_INDEX])

    # 3. The repository's canonical dependency set (fastapi, qwen-tts, kokoro,
    #    chatterbox, transformers<=4.57.6, librosa, ...).
    _run(["install", "-r", str(REPO_ROOT / "backend" / "requirements.txt")])

    # 4. Chatterbox / TADA pins that break the shared torch version — the exact
    #    same --no-deps treatment the justfile applies.
    _run(["install", "--no-deps", "chatterbox-tts"])
    _run(["install", "--no-deps", "hume-tada"])

    # 5. Qwen3-TTS library (git, as in the justfile).
    _run(["install", "git+https://github.com/QwenLM/Qwen3-TTS.git"])

    # 6. Point the HF cache at the models dir BEFORE importing backend.config,
    #    so every download (and the server later) uses the same cache.
    os.environ["VOICEBOX_MODELS_DIR"] = str(MODELS_DIR)
    os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Seed the Vivian preset profile through Voice Box's own ORM.
    #    Order matters: set_data_dir() BEFORE database import, exactly like
    #    backend/main.py does, so seeding hits the same sqlite file the server
    #    will use (--data-dir DATA_DIR).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from backend import config

    config.set_data_dir(DATA_DIR)

    from backend.database import SessionLocal, VoiceProfile, init_db

    init_db()
    db = SessionLocal()
    try:
        existing = db.get(VoiceProfile, VIVIAN_PROFILE_ID)
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
            print(f"Seeded profile {VIVIAN_PROFILE_ID} ({VIVIAN_NAME})")
        else:
            print(
                f"Profile {VIVIAN_PROFILE_ID} already present — "
                "not overwriting (preset_engine="
                f"{existing.preset_engine}, preset_voice_id={existing.preset_voice_id})"
            )
    finally:
        db.close()

    # 8. Pre-download the Qwen CustomVoice 1.7B weights so /generate/stream
    #    passes ensure_model_cached_or_raise() on first request.
    from huggingface_hub import snapshot_download

    print(f"\nPre-downloading {QWEN_CUSTOM_VOICE_1_7B} into {MODELS_DIR} ...")
    path = snapshot_download(
        repo_id=QWEN_CUSTOM_VOICE_1_7B,
        cache_dir=str(MODELS_DIR),
    )
    print(f"Model ready at: {path}")

    from qwen_tts import Qwen3TTSModel  # noqa: F401  (import smoke test)

    print("\nSetup complete. Start the server with:")
    print("  python kaggle/start_voicebox.py")
    print("Then verify:")
    print("  python kaggle/verify_voicebox.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())