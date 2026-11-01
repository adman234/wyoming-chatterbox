# TORCH_CUDA=cu128 covers Turing up to Blackwell (RTX 50xx). Pascal (GTX 10xx) needs --build-arg TORCH_CUDA=cu126.
FROM python:3.11-slim

ARG TORCH_CUDA=cu128
# torchvision and torchaudio must match torch exactly; Renovate updates all three in one PR.
# renovate: datasource=pypi depName=torch
ARG TORCH_VERSION=2.7.1
# renovate: datasource=pypi depName=torchvision
ARG TORCHVISION_VERSION=0.22.1
# renovate: datasource=pypi depName=torchaudio
ARG TORCHAUDIO_VERSION=2.7.1

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install torch stack first from the cuda wheel index, pinned to a known-good set
# torchvision must match torch exactly or the C++ nms operator won't exist
# cu128 is the first build with blackwell (rtx 50xx, sm_120) kernels; needs host driver >= 570
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    torchaudio==${TORCHAUDIO_VERSION} \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

# install chatterbox-tts with --no-deps to prevent it from clobbering the torch stack,
# then manually satisfy its remaining deps.
# pinned to an upstream commit: the pypi 0.1.7 release has no chatterbox-nano support.
# its remaining dependencies (and wyoming) are pinned in docker/requirements.txt
ARG CHATTERBOX_REF=5de7a54aa4e5e2baadb0182dde554908b48b85c2
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-deps "https://github.com/resemble-ai/chatterbox/archive/${CHATTERBOX_REF}.tar.gz" && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY pyproject.toml .
COPY wyoming_chatterbox/ ./wyoming_chatterbox/

# --no-deps again: resolving chatterbox-tts here would pull its torch==2.6.0 pin back in
RUN pip install --no-cache-dir --no-deps .

# fail the build if any later install swapped out the cuda torch stack
RUN python -c "import torch, torchvision, torchaudio; \
assert torch.__version__ == '${TORCH_VERSION}+${TORCH_CUDA}', torch.__version__; \
assert torchvision.__version__ == '${TORCHVISION_VERSION}+${TORCH_CUDA}', torchvision.__version__; \
assert torchaudio.__version__ == '${TORCHAUDIO_VERSION}+${TORCH_CUDA}', torchaudio.__version__"

# smoke test: the server and both model families import (no GPU or model download needed)
RUN python -c "import wyoming_chatterbox.__main__, chatterbox.tts, chatterbox.tts_turbo" && \
    wyoming-chatterbox --help > /dev/null

VOLUME ["/cache", "/voice"]

ENTRYPOINT ["wyoming-chatterbox"]
CMD ["--uri", "tcp://0.0.0.0:10800", "--voice-ref", "/voice/reference.wav"]
