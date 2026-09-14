FROM python:3.11-slim

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
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# install chatterbox-tts with --no-deps to prevent it from clobbering the torch stack,
# then manually satisfy its remaining deps.
# pinned to an upstream commit: the pypi 0.1.7 release has no chatterbox-nano support.
# chatterbox requires numpy<2; under numpy 2 promotion rules turbo's loudness
# normalisation turns the reference audio float64 and the s3 tokenizer rejects it
ARG CHATTERBOX_REF=5de7a54aa4e5e2baadb0182dde554908b48b85c2
RUN pip install --no-cache-dir --no-deps "https://github.com/resemble-ai/chatterbox/archive/${CHATTERBOX_REF}.tar.gz" && \
    pip install --no-cache-dir \
    "numpy<2" \
    omegaconf \
    librosa \
    s3tokenizer \
    pykakasi \
    conformer \
    safetensors \
    transformers==4.46.3 \
    pyloudnorm \
    spacy-pkuseg \
    resemble-perth \
    diffusers==0.29.0 \
    soundfile \
    "optree>=0.13.0"

COPY pyproject.toml .
COPY wyoming_chatterbox/ ./wyoming_chatterbox/

# --no-deps again: resolving chatterbox-tts here would pull its torch==2.6.0 pin back in
RUN pip install --no-cache-dir "wyoming>=1.5.0" && \
    pip install --no-cache-dir --no-deps .

# fail the build if any later install swapped out the cuda torch stack
RUN python -c "import torch, torchvision, torchaudio; \
assert torch.__version__ == '2.7.1+cu128', torch.__version__; \
assert torchvision.__version__ == '0.22.1+cu128', torchvision.__version__; \
assert torchaudio.__version__ == '2.7.1+cu128', torchaudio.__version__"

VOLUME ["/cache", "/voice"]

ENTRYPOINT ["wyoming-chatterbox"]
CMD ["--uri", "tcp://0.0.0.0:10800", "--voice-ref", "/voice/reference.wav"]
