# wyoming-chatterbox

[wyoming protocol](https://github.com/rhasspy/wyoming) server for [chatterbox tts](https://github.com/resemble-ai/chatterbox) with voice cloning.

clone any voice with a 10-30 second audio sample. integrates directly with home assistant as a tts provider.

## requirements

- nvidia gpu with 2gb+ vram (about 1-3gb used at runtime depending on `--model` and `--dtype`, see [gpu memory](#gpu-memory))
- cuda 12.8 capable host driver (≥570), needed for rtx 50xx (blackwell) support
- [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on host
- docker + docker compose v2

## docker (recommended)

### 1. configure

```bash
git clone https://github.com/sudoxreboot/wyoming-chatterbox
cd wyoming-chatterbox
cp .env.example .env
```

edit `.env`:

```env
WYOMING_PORT=10800          # host port — change if 10800 is taken
VOICE_REF_DIR=/path/to/dir  # directory containing your reference wav
VOICE_REF_FILE=reference.wav
VOLUME_BOOST=3.0
TORCH_DEVICE=cuda
```

### 2. build and run

```bash
docker compose build
docker compose up -d
```

first run downloads ~3.5gb of chatterbox model weights into a named docker volume (`chatterbox-cache`). this only happens once.

### 3. check logs

```bash
docker compose logs -f
# you should see: "starting server at tcp://0.0.0.0:10800"
```

### voice reference tips

- 10-30 seconds of clean speech
- no background music or noise
- consistent speaking style
- wav format (any sample rate)

---

## install from source (no docker)

```bash
git clone https://github.com/sudoxreboot/wyoming-chatterbox
cd wyoming-chatterbox
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

```bash
wyoming-chatterbox --uri tcp://0.0.0.0:10800 --voice-ref /path/to/voice.wav
```

### options

| option | default | description |
|--------|---------|-------------|
| `--uri` | required | server uri (e.g., `tcp://0.0.0.0:10800`) |
| `--voice-ref` | required | path to voice reference wav (10-30s of speech) |
| `--volume-boost` | 3.0 | output volume multiplier |
| `--device` | cuda | torch device (`cuda` or `cpu`) |
| `--model` | standard | `standard` (500M), `turbo` (350M, faster, supports `[laugh]` style tags) or `nano` (110M, fastest) |
| `--dtype` | float32 | precision for the T3 model: `float32`, `bfloat16` or `float16` |
| `--sdpa` | efficient | attention kernels pytorch may use: `efficient` (no flash/cudnn), `math` or `auto` (all) |
| `--s3gen-dtype` | float32 | precision for the s3gen flow (speech tokens to mel): `float32`, `bfloat16` or `float16`. the vocoder always stays float32 |
| `--debug` | false | enable debug logging |

---

## systemd service (source install)

```bash
sudo tee /etc/systemd/system/wyoming-chatterbox.service << EOF
[Unit]
Description=Wyoming Chatterbox TTS
After=network-online.target

[Service]
Type=simple
User=$(whoami)
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStart=$(pwd)/.venv/bin/wyoming-chatterbox \
  --uri tcp://0.0.0.0:10800 \
  --voice-ref /path/to/voice_reference.wav \
  --volume-boost 3.0
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now wyoming-chatterbox
```

---

## home assistant

1. settings → devices & services → add integration
2. search **wyoming protocol**
3. host: your server ip, port: `10800` (or whatever you set in `.env`)
4. select it as your tts provider in the voice assistant pipeline

---

## gpu memory

vram allocated by pytorch with the voice loaded (idle) and while speaking home assistant style sentences (peak), measured on an rtx 4070 super with torch 2.7.1 cu128. `nvidia-smi` will read a few hundred mb higher for the cuda context.

| `--model` | `--dtype bfloat16` idle / peak | `--dtype bfloat16 --s3gen-dtype bfloat16` idle / peak | speed (seconds per second of audio) |
|-----------|--------------------------------|--------------------------------------------------------|-------------------------------------|
| standard | 1.5gb / 1.8gb | 1.3gb / 1.6gb | ~0.4 |
| turbo | 1.3gb / 1.6gb | 1.1gb / 1.3gb | ~0.18 |
| nano | 0.9gb / 1.2gb | 0.65gb / 0.9gb | ~0.11 |

`--dtype float32` (the default) adds about 1.0gb for standard, 0.8gb for turbo and 0.35gb for nano.

- `--dtype` and `--s3gen-dtype` made no difference to whisper transcription accuracy or speaker similarity in testing. `bfloat16` needs an rtx 30xx or newer.
- `turbo` and `nano` are english only and ignore exaggeration/cfg, but support tags like `[laugh]` and `[cough]`.
- the speech tokenizer, speaker encoder and voice encoder are only needed to embed the reference voice, so they move to the cpu once it is loaded (about 500mb). changing the voice needs a restart.
- chatterbox keeps the speaker embedding's autograd graph alive (about 270mb of activations); it is detached after loading the voice.
- gpt-2 based models (turbo, nano) carry 0.75-1.5gb of unused attention mask buffers under transformers 4.x; these are freed at load.
- models are loaded and trimmed in system ram before moving to the gpu, so freed buffers and float32 copies never leave unreturnable holes in gpu memory. startup briefly uses 3-5gb of system ram.
- text is generated one sentence at a time, so long announcements do not raise peak memory. unused cached memory is released after every request, and memory does not grow across repeated requests.
- flash/cudnn attention made `nano` with `bfloat16` speak gibberish on an rtx 5060 ti, so they are off by default (`--sdpa efficient`). `--sdpa auto` turns them back on.

if you get oom errors:

```bash
nvidia-smi

# docker
docker compose restart

# source
pkill -f wyoming-chatterbox
```

---

## license

mit

---

<div align="center">

made by [sudoxnym](https://sudoxreboot.com) ⚡

</div>
