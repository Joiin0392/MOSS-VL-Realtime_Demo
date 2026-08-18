# MOSS-VL Realtime Demo App

Realtime voice + video chat demo around the MOSS-VL streaming VLM. A FastAPI
gateway (`server/`) orchestrates VLM workers, ASR (FunASR SenseVoice) and TTS
engines behind one WebSocket plane; a React/Vite frontend (`src/`) serves the
live chat page (camera preview, captions, voice turns).

The demo runs as one tmux session with two processes:

- `api` — uvicorn on `:8000`; spawns and health-gates its own VLM workers and
  TTS sidecars in the FastAPI lifespan (`server/sidecars.py`, `server/gpu/supervisor.py`).
- `web` — `vite build --watch` + `vite preview` on `:20941`.

## File structure

```
server/            FastAPI gateway + orchestration (sessions, GPU supervisor, sidecars)
  adapters/        ASR / VLM / TTS provider adapters (vendored sidecar code under tts/*/sidecar/third_party/)
  gpu/             GPU placement planning + VLM worker supervisor
  session/         session manager + turn orchestrator
  realtime/        WebSocket realtime plane
src/               React + TypeScript frontend (Vite)
scripts/
  build_venv*.sh   venv builders (one per env; see below)
  deploy/demo.sh   ONE deploy entrypoint: up / down / restart / status / doctor / logs
  deploy/run_backend.sh, run_web.sh   the two demo processes
requirements*.txt  per-venv pins (see table above)
models/            model zoo (gitignored, symlinks) — asr/ tts/ vlms/
.env.deploy.example  box-local deploy overrides (copy to .env.deploy)
```

## Quickstart

Target box: Linux GPU box (developed on H200, driver CUDA 12.9), Python 3.12,
Node ≥ 20.19.

### 1. Model checkpoints

The backend expects this zoo layout under `models/` (every path overridable via
`.env.deploy` / env — see `server/config.py`):

```
models/
  asr/SenseVoiceSmall                iic/SenseVoiceSmall (ModelScope)
  asr/fsmn-vad                       iic/speech_fsmn_vad_zh-cn-16k-common-pytorch (ModelScope)
  tts/MOSS-TTS-Nano-100M             OpenMOSS-Team/MOSS-TTS-Nano-100M
  tts/MOSS-Audio-Tokenizer-Nano      OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano
  tts/MOSS-TTS-Realtime              OpenMOSS-Team/MOSS-TTS-Realtime        (moss_tts_realtime)
  tts/MOSS-Audio-Tokenizer           OpenMOSS-Team/MOSS-Audio-Tokenizer     (moss_tts_realtime)
  tts/Fun-CosyVoice3-0.5B-2512       FunAudioLLM/Fun-CosyVoice3-0.5B-2512   (cosyvoice3 via vllm)
  vlms/hf_mossvl_streaming_processor online streaming MOSS-VL checkpoint (internal)
  vlms/offline                       offline MOSS-VL checkpoint (internal, optional)
```

Download the public ones (e.g. `pip install huggingface_hub modelscope`):

```bash
hf download OpenMOSS-Team/MOSS-TTS-Nano-100M        --local-dir models/tts/MOSS-TTS-Nano-100M
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano --local-dir models/tts/MOSS-Audio-Tokenizer-Nano
hf download OpenMOSS-Team/MOSS-TTS-Realtime         --local-dir models/tts/MOSS-TTS-Realtime
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer      --local-dir models/tts/MOSS-Audio-Tokenizer
hf download FunAudioLLM/Fun-CosyVoice3-0.5B-2512    --local-dir models/tts/Fun-CosyVoice3-0.5B-2512
modelscope download --model iic/SenseVoiceSmall                              --local_dir models/asr/SenseVoiceSmall
modelscope download --model iic/speech_fsmn_vad_zh-cn-16k-common-pytorch     --local_dir models/asr/fsmn-vad
```

The two `models/vlms/` checkpoints are internal SFT artifacts — place or
symlink them there yourself. Without `vlms/offline`, the offline-chat sidecar
is skipped and chat falls back to the online pool.

### 2. Build the venvs

```bash
bash scripts/build_venv.sh          # .venv      — gateway + VLM + ASR (~10-30 min, builds flash-attn)

python3.12 -m venv .venv-vllm       # .venv-vllm — default TTS engines (see requirements-vllm.txt header
PIP_CONSTRAINT=/dev/null .venv-vllm/bin/pip install \   #  for the offline-wheelhouse variant)
    -r requirements-vllm.txt
.venv-vllm/bin/python scripts/patch_vllm_omni_moss_codec.py   # required for moss_tts_realtime

bash scripts/build_venv_sglang.sh   # .venv-sglang — offline chat (optional; needs the sglang fork checkout)
bash scripts/build_venv_cosyvoice.sh  # .venv-cosy  — CosyVoice3 native sidecar (optional)
bash scripts/build_venv_mossrt.sh     # .venv-mossrt — MOSS-TTS-Realtime native sidecar (optional)
```

Only `.venv` + `.venv-vllm` are needed for the default demo; the rest unlock
optional providers (`OFFLINE_PROVIDER`, `TTS_PROVIDER=*_native`). The build
scripts default to an internal PyPI mirror / wheelhouse — set `MIRROR=...` or
`WHEELHOUSE=...` to adapt (see each script's header).

### 3. Frontend

```bash
npm install
npm run build
```

### 4. Run

```bash
scripts/deploy/demo.sh up       # everything up (api :8000, web :20941)
scripts/deploy/demo.sh status   # health at a glance
scripts/deploy/demo.sh logs api # follow logs
scripts/deploy/demo.sh down     # tear down
```

Open `http://<box>:20941`. First boot is slow (~10-20 min cold): the backend
health-gates the VLM load and TTS sidecars before answering. Box-local
overrides (ports, model paths, providers) go in `.env.deploy` — see
`.env.deploy.example`.
