# Voice Box on a Kaggle GPU

Run the **Voice Box** server (FastAPI, `backend/main:app`) on a Kaggle **GPU
notebook (T4)** so the Survée AI Waiter can synthesize **Vivian** (Qwen
CustomVoice, 1.7B, preset profile `d69609bd-1117-4a39-8d9c-1f57368b8835`) on a
CUDA card.

This directory is **additive only** — it deploys the existing repository, it
does not fork or re-implement anything:

- `setup_voicebox.py`  — installs deps the exact way the `justfile` does,
  seeds the Vivian preset row through the repo's own ORM, and pre-downloads the
  Qwen CustomVoice weights.
- `start_voicebox.py`  — starts the real `python -m backend.main` on
  `0.0.0.0:17493`.
- `verify_voicebox.py` — environment + GPU + health + profile + real
  cold/warm TTS verification, returns 0 = KAGGLE READY.

Nothing here changes local Voice Box behavior, `data/`, `.env`, the Survée
integration, or the `/generate/stream` API.

---

## Architecture

```
Kaggle GPU notebook (this server)          Survée (Hostinger, unchanged)
┌────────────────────────────────────┐      ┌──────────────────────────────┐
│ Voice Box  :17493  (0.0.0.0)      │      │ Next.js                       │
│   backend/main:app                 │◄─────│ app/api/voice/tts/route.ts    │
│   qwen_custom_voice (Vivian 1.7B)  │ POST │ lib/ai/voice/server/*         │
│   sqlite data/ voicebox.db         │      │  profile_id = d69609bd…8835   │
│   HF cache: Qwen3-TTS-12Hz-1.7B-   │─────►│  {text, language, profile_id} │
│     CustomVoice (bf16)             │ WAV  │ Browser plays /api/voice/tts  │
└────────────────────────────────────┘      └──────────────────────────────┘
```

- The browser never calls Voice Box directly. All traffic is Ship It server to
  Voice Box (CORS only matters if you browse the notebook tunnel).
- **Do not run the Next.js app in Kaggle.** Hostinger keeps serving Survée;
  Kaggle only renders audio.
- This phase has **no tunnel**: we prove the GPU pipeline first, then pick a
  tunnel (ngrok/Cloudflare/...).

### The integration contract Voice Box must satisfy (unchanged)

The Survée server (`lib/ai/voice/server/voicebox-provider.ts`) with default
config calls:

```
POST {base}/generate/stream
{"text": "<reply>", "language": "en", "profile_id": "d69609bd-1117-4a39-8d9c-1f57368b8835"}
Accept: audio/wav
```

- Default `base` is `http://127.0.0.1:17493` (see `VOICEBOX_BASE_URL`).
- **No `engine`, no `model_size`** are sent. Voice Box resolves both **from the
  profile row**: `preset_engine/qwen_custom_voice`, `preset_voice_id/Vivian`,
  and the `1.7B` default in `generations.py` (`model_size or "1.7B"`).
- The endpoint requires the model to be **already cached** (`ensure_model_cached_or_raise`
  -> 400 message "Model ... is not downloaded yet") and the profile row to exist
  (else 404 "Profile not found"). That is exactly what `setup_voicebox.py` prepares.
- Timeout the Survée side tolerates: `VOICEBOX_TIMEOUT_MS`, default 120 000 ms.

---

## One-time setup (Kaggle GPU notebook)

GPU accelerator **must** be enabled (T4 / P100 / T4 x2). Session accelerator:
`GPU > T4 x2` (or T4 x1).

```python
# 1. Clone Voice Box (API-only; there is no frontend/ tree to build).
!git clone <YOUR-VOICEBOX-REPO-GIT-URL> /kaggle/working/voicebox
%cd /kaggle/working/voicebox

# 2. Install deps, seed Vivian, pre-download the 1.7B CustomVoice weights
#    (torch cu128 + backend/requirements.txt + Qwen3-TTS git, exactly like
#    the justfile Linux/NVIDIA branch).
!python kaggle/setup_voicebox.py

# 3. Start the server in a background process (keep it alive for this session).
import subprocess
subprocess.Popen(
    ["python", "kaggle/start_voicebox.py"],
    stdout=open("/kaggle/working/voicebox.log", "w"),
    stderr=subprocess.STDOUT,
)

# 4. Verify end-to-end (returns 0 = KAGGLE READY).
!python kaggle/verify_voicebox.py
```

Expected verify output (T4):

```
1. Python / PyTorch / CUDA environment
  torch            : 2.x.X+cu128
  cuda available   : True
  gpu[0]           : Tesla T4 (compute 7.5, SM 75, total 15.8 GiB, ...)
2. Voice Box API health
  gpu_available    : True
  gpu_type         : CUDA (Tesla T4)
3. Vivian preset profile
  preset_engine    : qwen_custom_voice
  preset_voice_id  : Vivian
4. End-to-end TTS
  [COLD] ttfb ~=(model load + first gen, typically 15-90 s on T4)
  [WARM] ttfb ~= (generation only)
OVERALL: KAGGLE READY
```

Notes:

- Kaggle sessions are **ephemeral**. When the notebook is reset, re-run the
  setup cell — pip wheels and the HF cache re-download. To persist the ~3.5 GB
  of weights across sessions, publish the cached HF repo as a private dataset
  (see "Persisting the model cache" below).
- If `cuda available : False` after setup, the torch build and the Kaggle
  driver disagree. Retry with a different wheel index:
  `!VOICEBOX_TORCH_INDEX=https://download.pytorch.org/whl/cu124 python kaggle/setup_voicebox.py`
  (cu124 / cu126 / cu128 all include sm_75 for T4).

---

## Routines

| Task | Command |
|---|---|
| Install + seed + pre-download | `python kaggle/setup_voicebox.py` |
| Start server (foreground) | `python kaggle/start_voicebox.py` |
| Start server (notebook, background) | see cell 3 above |
| Verify ready (exit 0/1/2) | `python kaggle/verify_voicebox.py` |
| Health probe | `curl http://127.0.0.1:17493/health` |
| Ad-hoc TTS | `curl -X POST http://127.0.0.1:17493/generate/stream -H "Content-Type: application/json" -d '{"text":"Hello","language":"en","profile_id":"d69609bd-1117-4a39-8d9c-1f57368b8835"}' -o hello.wav` |

---

## What is verified

- **GPU**: `torch.cuda.is_available()`, device count, device name + compute
  capability, VRAM, and `check_cuda_compatibility()` (build vs `sm_XX`).
- **Health**: `/health` reports `gpu_available` and `gpu_type` (CUDA T4).
- **Vivian**: the exact Survée UUID resolves to
  `preset / qwen_custom_voice / Vivian`.
- **TTS**: the **exact Survée payload** against `/generate/stream`, twice —
  cold (model load) then warm (no restart) — measuring **TTFB**, total time,
  and validating the returned WAV (RIFF/WAVE, sample rate, channels, duration).

Enable the GPU in the notebook **before** running setup: `Settings >
Accelerator > GPU (T4)`.

---

## Latency methodology (record these in the report)

Send the exact production payload; measure with `Popen`/`httpx` timestamps:

- `T0` — request transmitted.
- `TTFB` — first WAV byte received (this endpoint assembles audio before
  streaming, so TTFB ≈ generation time).
- `TOTAL` — stream fully received.
- **COLD** run = the first request after the server starts (includes loading
  the 1.7B CustomVoice model into VRAM).
- **WARM** run = an immediate second request on the same process (model reused).

A T4 is ~4-8x faster than a CPU for the 1.7B CustomVoice; expect cold TTFB in
the tens of seconds and warm TTFB well under `VOICEBOX_TIMEOUT_MS` (120 s).

---

## Persisting the model cache across Kaggle sessions (optional)

`/kaggle/working` is wiped on session reset, so the ~3.5 GB weights re-download.
To avoid that:

1. After setup, `!mkdir -p /kaggle/working/voicebox-models` already holds
   `models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice`.
2. Upload `/kaggle/working/voicebox-models` as a private Kaggle **Dataset**
   (or a private HF model tree).
3. Next session: `!kaggle datasets download` / `!cp` it back into place, or set
   `VOICEBOX_MODELS_DIR` to its path on `/kaggle/input` and skip the big
   download in setup (setup skips already-cached repos automatically).

---

## Known differences vs local/Docker (by design)

| Aspect | Local / Docker | Kaggle |
|---|---|---|
| Bind host | `127.0.0.1:17493` | `0.0.0.0:17493` (tunnel-ready) |
| Data dir | `./data` (or `--data-dir`) | `--data-dir /kaggle/working/voicebox-data` |
| HF cache | default | `VOICEBOX_MODELS_DIR=/kaggle/working/voicebox-models` |
| torch | cu128 (NVIDIA) | same cu128 default, overridable |
| GPU | none/CPU | CUDA T4 (sm_75, bf16) |

Application code is identical (`python -m backend.main`). No local files,
`.env`, profile, or route are changed.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cuda available: False` | torch/driver mismatch. Retry setup with `VOICEBOX_TORCH_INDEX=.../cu124` (or cu126), or confirm the accelerator is a GPU, not CPU. |
| `HTTP 400 Model 1.7B is not downloaded yet` | setup did not finish the HF pre-download (network). Re-run `kaggle/setup_voicebox.py`. |
| `HTTP 404 Profile not found` | setup was not run (profile row missing). Re-run setup so the DB is seeded. |
| COLD request very slow | expected: bf16 1.7B load + first inference on T4. WARM must be fast. |
| `KAGGLE BLOCKED` from verify | every failing check is printed; fix per check, re-verify. |
| Session reset killed the server | Kaggle notebooks are ephemeral — re-run cells 1-3. |

---

## Boundaries (do not break these)

- No change to local Voice Box runtime, `data/`, or `.env`.
- No change to Vivian's identity: same UUID, same preset values.
- No change to `/generate/stream` semantics or the Survée payload.
- No tunnel added in this phase; the Survée `VOICEBOX_BASE_URL` stays at its
  current value until a tunnel exists.