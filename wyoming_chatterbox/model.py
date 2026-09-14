"""Chatterbox model loading and precision handling."""

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


def load_model(name: str, device: str, dtype: str = "float32"):
    """Load a Chatterbox variant, optionally running T3 in lower precision."""
    if name == "standard":
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)
    elif name in ("turbo", "nano"):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        kwargs = {"nano": True} if name == "nano" else {}
        model = ChatterboxTurboTTS.from_pretrained(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

    if getattr(model.t3, "is_gpt", False):
        drop_causal_mask_buffers(model.t3.tfmr)

    torch_dtype = DTYPES[dtype]
    if torch_dtype != torch.float32:
        cast_t3(model.t3, torch_dtype)
        _LOGGER.info("T3 running in %s", dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


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
