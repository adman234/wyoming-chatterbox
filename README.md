# wyoming-chatterbox

A [Wyoming protocol](https://github.com/rhasspy/wyoming) server for
[Chatterbox TTS](https://github.com/resemble-ai/chatterbox), so Home Assistant can speak
in a voice cloned from a 10 to 30 second audio sample.

This is a fork of [sudoxreboot/wyoming-chatterbox](https://github.com/sudoxreboot/wyoming-chatterbox)
by [sudoxnym](https://sudoxreboot.com), who wrote the Wyoming server. This fork packages
it for Unraid and cuts its GPU memory use. It is not intended to be merged back upstream.

## What is different from upstream

- **Prebuilt image** at `ghcr.io/adman234/wyoming-chatterbox`, and an Unraid template.
- **PyTorch 2.7.1 with CUDA 12.8**, so RTX 50 series (Blackwell) cards work. The host
  needs NVIDIA driver 570 or newer.
- **Smaller models and lower precision**: `--model turbo` or `nano`, and `--dtype` /
  `--s3gen-dtype` for bfloat16 or float16. Together these take it from about 3 GB of
  VRAM to under 1 GB. See [GPU memory](#gpu-memory).
- **VRAM trimming**: models are loaded and trimmed on the CPU before moving to the GPU,
  encoders only needed for the reference voice move back to the CPU, and unused buffers
  are freed.
- **`--sdpa`** to choose attention kernels. Flash and cuDNN attention are off by default
  because they made `nano` produce gibberish on an RTX 5060 Ti.

## Install

### Unraid

Fetch the template from the Unraid terminal, then go to Docker > Add Container and pick
`wyoming-chatterbox` from the template list. It needs the Nvidia Driver plugin.

```bash
wget -O /boot/config/plugins/dockerMan/templates-user/my-wyoming-chatterbox.xml https://raw.githubusercontent.com/adman234/wyoming-chatterbox/master/unraid/wyoming-chatterbox.xml
```

Put your reference WAV in the Voice Reference folder. Its filename must match
`--voice-ref` in Post Arguments (`/voice/reference.wav` by default). Extra options such
as `--model turbo --dtype bfloat16` also go in Post Arguments.

### docker compose

```bash
git clone https://github.com/adman234/wyoming-chatterbox
cd wyoming-chatterbox
cp .env.example .env    # set VOICE_REF_DIR and VOICE_REF_FILE
docker compose up -d --build
```

The first start downloads 3 to 4 GB of model weights into the cache volume. The log
shows `starting server at tcp://0.0.0.0:10800` when it is ready.

### From source

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install .
wyoming-chatterbox --uri tcp://0.0.0.0:10800 --voice-ref /path/to/voice.wav
```

## Home Assistant

1. Settings > Devices & services > Add integration > **Wyoming Protocol**.
2. Host: your server's IP. Port: `10800`.
3. Choose it as the TTS provider in your voice assistant pipeline.

For a good clone, use 10 to 30 seconds of clean speech in a WAV file, with no music or
background noise and a consistent speaking style.

## Options

| Option | Default | Description |
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

## GPU memory

VRAM allocated by PyTorch with the voice loaded (idle) and while speaking home assistant style sentences (peak), measured on an rtx 4070 super with torch 2.7.1 cu128. `nvidia-smi` will read a few hundred mb higher for the cuda context.

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

If you run out of GPU memory, restart the container.

## License

MIT, as upstream.
