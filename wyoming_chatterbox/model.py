"""Chatterbox model loading, precision and VRAM handling."""

import logging

import torch

_LOGGER = logging.getLogger(__name__)

MODELS = ("standard", "turbo", "nano")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
SDPA_BACKENDS = ("auto", "efficient", "math")

# approximate size of the Hugging Face snapshot each model downloads on first start
DOWNLOAD_GB = {"standard": 3.2, "turbo": 4.0, "nano": 3.0}


def configure_sdpa(backend: str) -> None:
    """Restrict which scaled dot product attention kernels PyTorch may pick.

    "efficient" disables the flash and cuDNN kernels, "math" also disables the
    memory efficient kernel. The flags are process wide, so they also apply to
    generations running in executor threads.
    """
    if backend != "auto":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    if backend == "math":
        torch.backends.cuda.enable_mem_efficient_sdp(False)

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        _LOGGER.info(
            "GPU %s (sm_%d%d), torch %s, SDPA kernels: flash=%s cudnn=%s efficient=%s math=%s",
            torch.cuda.get_device_name(),
            major,
            minor,
            torch.__version__,
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )


def load_model(name: str, device: str, dtype: str = "float32", s3gen_dtype: str = "float32"):
    """Load a Chatterbox variant, optionally running T3 and the S3Gen flow in lower precision."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")

    _LOGGER.info(
        "Loading %s model files (the first start downloads about %.1f GB, "
        "which can take a while with no progress shown)...",
        name,
        DOWNLOAD_GB[name],
    )
    if name == "standard":
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        kwargs = {"nano": True} if name == "nano" else {}
        model = ChatterboxTurboTTS.from_pretrained(device=device, **kwargs)

    # S3Gen derives its device and dtype from the speech tokenizer's parameters.
    # The tokenizer moves to the CPU after voice preparation and the flow may run
    # in half precision, so read them from the float32 vocoder, which never moves
    # or changes dtype. Voice preparation (fbank, FFT) needs float32.
    s3gen_cls = type(model.s3gen)
    s3gen_cls.device = property(lambda self: next(self.mel2wav.parameters()).device)
    s3gen_cls.dtype = property(lambda self: next(self.mel2wav.parameters()).dtype)

    if getattr(model.t3, "is_gpt", False):
        drop_causal_mask_buffers(model.t3.tfmr)

    torch_dtype = DTYPES[dtype]
    if torch_dtype != torch.float32:
        cast_t3(model.t3, torch_dtype)
        _LOGGER.info("T3 running in %s", dtype)

    flow_dtype = DTYPES[s3gen_dtype]
    if flow_dtype != torch.float32:
        cast_s3gen_flow(model.s3gen, flow_dtype)
        _LOGGER.info("S3Gen flow running in %s", s3gen_dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def prepare_voice(model, voice_ref: str) -> None:
    """Embed the reference voice, then free everything only that step needed.

    Voice preparation runs the S3 speech tokenizer, the CAMPPlus speaker
    encoder and the voice encoder. Generation never uses them again, so they
    move to the CPU (about 500 MiB). The CAMPPlus embedding also comes back
    with its autograd graph attached, which keeps about 270 MiB of activations
    alive, so it is detached. Changing the voice afterwards needs a restart.
    """
    model.prepare_conditionals(voice_ref)

    gen = model.conds.gen
    for key in gen:
        if torch.is_tensor(gen[key]):
            gen[key] = gen[key].detach()

    # safe because load_model pointed S3Gen's device and dtype at the vocoder
    model.s3gen.tokenizer.to("cpu")
    model.s3gen.speaker_encoder.to("cpu")
    model.ve.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def drop_causal_mask_buffers(tfmr: torch.nn.Module) -> None:
    """Free the per-layer causal mask buffers of a GPT-2 backbone.

    transformers 4.x registers a (1, 1, n_positions, n_positions) bool mask in
    every GPT-2 attention layer. Turbo and Nano use n_positions=8196, so that is
    64 MiB per layer: 1.5 GiB for Turbo, 0.75 GiB for Nano. The SDPA attention
    path never reads it; only the eager fallback (output_attentions or head_mask)
    does, and with the buffer gone that fails loudly instead of running.
    transformers 5 no longer has the buffer.
    """
    if tfmr.config._attn_implementation != "sdpa":
        return

    freed = 0
    for module in tfmr.modules():
        mask = module._buffers.get("bias")
        if mask is not None and mask.dtype == torch.bool:
            freed += mask.numel() * mask.element_size()
            del module._buffers["bias"]

    if freed:
        _LOGGER.info("Freed %.0f MiB of unused GPT-2 causal mask buffers", freed / 2**20)


def cast_t3(t3: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast T3 (the text to speech token model) to dtype.

    Chatterbox has no dtype option. The speaker embedding and emotion value
    come from float32 encoders, so they are cast on the way into the
    conditioning encoder, and speech logits are cast back to float32 so
    CFG, top-p and sampling stay in full precision.
    """
    t3.to(dtype=dtype)

    def cast_cond(_module, args):
        # T3Cond.to leaves integer tensors (prompt speech tokens) alone
        args[0].to(dtype=dtype)

    t3.cond_enc.register_forward_pre_hook(cast_cond)
    t3.speech_head.register_forward_hook(lambda _module, _args, out: out.float())


def cast_s3gen_flow(s3gen: torch.nn.Module, dtype: torch.dtype) -> None:
    """Run the S3Gen flow (speech tokens to mel spectrogram) in dtype.

    The CFM decoder already casts its inputs to its own dtype, but the conformer
    encoder does not, so the call runs under autocast and the mels come back as
    float32. The HiFT vocoder stays in float32 because its STFT needs it.
    """
    flow = s3gen.flow
    flow.to(dtype=dtype)
    device_type = next(flow.parameters()).device.type
    inference = flow.inference

    def autocast_inference(*args, **kwargs):
        with torch.autocast(device_type=device_type, dtype=dtype):
            out = inference(*args, **kwargs)
        return tuple(o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in out)

    flow.inference = autocast_inference
