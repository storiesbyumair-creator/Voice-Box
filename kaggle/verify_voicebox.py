"""Kaggle GPU readiness verifier for Voice Box.

Checks, in order:

  1. Python / PyTorch / CUDA environment (GPU present, compute capability,
     VRAM, compatibility with this torch build).
  2. Voice Box is up: GET /health — gpu_available, gpu_type, backend_type.
  3. Survée's Vivian preset profile exists (exact UUID) and routes to
     qwen_custom_voice.
  4. Real end-to-end TTS over the exact Survée payload
     ({text, language: "en", profile_id}) to POST /generate/stream:
       - COLD  = first request (loads the 1.7B CustomVoice model)
       - WARM  = second request, no restart (reuses the loaded model)
     Latency is measured as TTFB (first byte) and total, and the WAV is
     validated (RIFF/WAVE header, sample rate, channels, duration).

Exit codes:
  0  KAGGLE READY   - GPU + server + profile + TTS all verified.
  1  KAGGLE BLOCKED - one or more hard checks failed (see output).
  2  could not reach the server at all.

Usage:
  python kaggle/verify_voicebox.py

Env overrides:
  VOICEBOX_BASE_URL  default http://127.0.0.1:17493
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import wave

BASE_URL = os.environ.get("VOICEBOX_BASE_URL", "http://127.0.0.1:17493")
VIVIAN_PROFILE_ID = "d69609bd-1117-4a39-8d9c-1f57368b8835"
TEST_TEXT = "Hello, welcome to our restaurant. How can I help you today?"

COLD_TIMEOUT_S = 900  # model load + first generation on a T4
WARM_TIMEOUT_S = 180


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def fail(message: str) -> int:
    print(f"\nFAIL: {message}")
    return 1


def section_gpu() -> tuple[bool, str]:
    banner("1. Python / PyTorch / CUDA environment")
    try:
        import torch
    except ImportError as e:
        return False, f"torch not importable ({e}). Run kaggle/setup_voicebox.py first."

    print(f"  python           : {sys.version.split()[0]}")
    print(f"  torch            : {torch.__version__}")
    print(f"  torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
    print(f"  torch build arch : {getattr(torch.cuda, '_get_arch_list', lambda: None)() or 'n/a'}")
    print(f"  cuda available   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False — no usable GPU for this torch build."

    count = torch.cuda.device_count()
    print(f"  gpu count        : {count}")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        total_bytes, free_bytes = torch.cuda.mem_get_info(i)
        print(
            f"  gpu[{i}]          : {name} (compute {major}.{minor}, "
            f"SM {major}{minor}, total {total_bytes / 1024 ** 3:.1f} GiB, "
            f"free {free_bytes / 1024 ** 3:.1f} GiB)"
        )

    from backend.backends.base import check_cuda_compatibility

    compatible, warning = check_cuda_compatibility()
    if warning:
        print(f"  compatibility   : WARNING -> {warning}")
        return False, f"GPU not supported by this PyTorch build: {warning}"
    print("  compatibility   : OK (this build supports the detected GPU)")
    return True, None


def section_health() -> tuple[bool, dict | None]:
    banner("2. Voice Box API health")
    import httpx

    url = f"{BASE_URL}/health"
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
        ok = resp.status_code == 200
        if not ok:
            print(f"  {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
            return False, None
        data = resp.json()
        print(f"  status            : {data.get('status')}")
        print(f"  gpu_available     : {data.get('gpu_available')}")
        print(f"  gpu_type          : {data.get('gpu_type')}")
        print(f"  backend_type      : {data.get('backend_type')}")
        print(f"  backend_variant   : {data.get('backend_variant')}")
        print(f"  model_loaded      : {data.get('model_loaded')} (default qwen base engine)")
        print(f"  model_size        : {data.get('model_size')}")
        print(f"  vram_used_mb      : {data.get('vram_used_mb')}")
        print(f"  gpu_compat_warning: {data.get('gpu_compatibility_warning')}")
        if not data.get("gpu_available"):
            return False, "/health reports gpu_available=False on a GPU runtime."
        return True, data
    except httpx.HTTPError as e:
        print(f"  {url} -> unreachable: {e}")
        return False, None


def section_profile() -> tuple[bool, dict | None]:
    banner("3. Vivian preset profile (Survée default profile_id)")
    import httpx

    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{BASE_URL}/profiles")
        if resp.status_code != 200:
            return False, f"GET /profiles -> HTTP {resp.status_code}: {resp.text[:200]}"
        profiles = resp.json()

    target = next((p for p in profiles if p.get("id") == VIVIAN_PROFILE_ID), None)
    if target is None:
        ids = ", ".join(p.get("id", "?") for p in profiles) or "(none)"
        return False, (
            f"profile {VIVIAN_PROFILE_ID} not found. Seeded DB? Run "
            f"kaggle/setup_voicebox.py. Present profiles: {ids}"
        )
    print(f"  id             : {target.get('id')}")
    print(f"  name           : {target.get('name')}")
    print(f"  voice_type     : {target.get('voice_type')}")
    print(f"  preset_engine  : {target.get('preset_engine')}")
    print(f"  preset_voice_id: {target.get('preset_voice_id')}")
    print(f"  default_engine : {target.get('default_engine')}")
    print(f"  language       : {target.get('language')}")

    engine_ok = target.get("voice_type") == "preset"
    route_ok = target.get("preset_engine") == "qwen_custom_voice"
    voice_ok = target.get("preset_voice_id") == "Vivian"
    if not (engine_ok and route_ok and voice_ok):
        return False, (
            "Vivian profile is not routed to qwen_custom_voice/Vivian. "
            f"voice_type={target.get('voice_type')}, "
            f"preset_engine={target.get('preset_engine')}, "
            f"preset_voice_id={target.get('preset_voice_id')}"
        )
    return True, target


def validate_wav(audio: bytes) -> dict:
    """Parse and sanity-check a WAV byte stream."""
    if len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file ({len(audio)} bytes, " f"magic {audio[:12]!r})")
    with wave.open(io.BytesIO(audio)) as wav:
        channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        nframes = wav.getnframes()
    duration = nframes / framerate if framerate else 0.0
    return {
        "channels": channels,
        "sample_width_bytes": sampwidth,
        "sample_rate": framerate,
        "frames": nframes,
        "duration_s": round(duration, 2),
    }


def run_tts(label: str, timeout_s: int) -> tuple[bool, str]:
    """POST the exact Survée payload to /generate/stream, measuring TTFB + total."""
    import httpx

    payload = {"text": TEST_TEXT, "language": "en", "profile_id": VIVIAN_PROFILE_ID}
    url = f"{BASE_URL}/generate/stream"
    started = time.monotonic()
    ttfb_ms = None
    status = None
    content_type = None
    audio = bytearray()

    try:
        with httpx.stream(
            "POST", url, json=payload, timeout=httpx.Timeout(timeout_s, connect=30)
        ) as resp:
            status = resp.status_code
            content_type = resp.headers.get("content-type")
            for chunk in resp.iter_bytes():
                if ttfb_ms is None:
                    ttfb_ms = (time.monotonic() - started) * 1000
                audio.extend(chunk)
        total_ms = (time.monotonic() - started) * 1000
    except httpx.HTTPError as e:
        print(f"  [{label}] HTTP error: {e}")
        return False, e

    elapsed = (time.monotonic() - started) * 1000
    print(f"  [{label}] status     : {status}")
    print(f"  [{label}] content-type: {content_type}")
    print(f"  [{label}] ttfb       : {ttfb_ms:.0f} ms" if ttfb_ms is not None else "  ttfb n/a")
    print(f"  [{label}] total      : {total_ms:.0f} ms")
    print(f"  [{label}] bytes      : {len(audio)}")

    if status != 200:
        return False, f"[{label}] HTTP {status}: {bytes(audio)[:300]!r}"

    try:
        wav = validate_wav(bytes(audio))
    except ValueError as e:
        return False, f"[{label}] WAV validation failed: {e}"

    print(f"  [{label}] wav        : {wav}")

    expected_rate = wav["sample_rate"]
    expected_channels = wav["channels"]
    if not (0 < expected_rate <= 96000 and expected_channels in (1, 2)):
        return False, f"[{label}] implausible wav metadata: {wav}"

    return True, None


def section_tts() -> tuple[bool, list[str]]:
    banner("4. End-to-end TTS — Survée payload (Vivian -> /generate/stream)")
    print(f"  test text : {TEST_TEXT!r}")
    print(f"  payload   : {{text, language: 'en', profile_id: {VIVIAN_PROFILE_ID}}}")
    print("  (cold = first request / model load, warm = second request, no restart)")
    errors: list[str] = []

    ok_cold, err_cold = run_tts("COLD", COLD_TIMEOUT_S)
    if not ok_cold:
        errors.append(err_cold or "cold TTS failed")

    ok_warm, err_warm = run_tts("WARM", WARM_TIMEOUT_S)
    if not ok_warm:
        errors.append(err_warm or "warm TTS failed")

    return (not errors, errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    checks: list[str] = []

    ok_gpu, err_gpu = section_gpu()
    checks.append("GPU/CUDA" + (" OK" if ok_gpu else f" FAIL ({err_gpu})"))

    ok_health, health = section_health()
    if health is None and not ok_health:
        # Bare httpx transport failure => server not running.
        print(
            "\nServer not reachable. Start it first, e.g.:\n"
            "  python kaggle/start_voicebox.py\n"
            "(in a notebook: run it as a background process, see the file docstring)"
        )
        return 2
    checks.append("health" + (" OK" if ok_health else f" FAIL ({health})"))

    ok_profile, profile = section_profile()
    checks.append("Vivian profile" + (" OK" if ok_profile else f" FAIL ({profile})"))

    ok_tts, tts_errors = section_tts()
    checks.append("TTS cold+warm" + (" OK" if ok_tts else f" FAIL ({', '.join(tts_errors)})"))

    banner("SUMMARY")
    for c in checks:
        print(f"  {c}")

    all_ok = ok_gpu and ok_health and ok_profile and ok_tts
    if all_ok:
        print("\nOVERALL: KAGGLE READY - Voice Box on GPU serves Vivian /generate/stream.")
        return 0

    blockers = [c for c in checks if "FAIL" in c]
    print(f"\nOVERALL: KAGGLE BLOCKED - {len(blockers)} blocker(s):")
    for c in blockers:
        print(f"  - {c}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())