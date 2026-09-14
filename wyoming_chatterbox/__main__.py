#!/usr/bin/env python3
"""Wyoming server for Chatterbox TTS."""

import argparse
import asyncio
import logging
from functools import partial
from pathlib import Path

from wyoming.server import AsyncServer

from . import __version__
from .handler import ChatterboxEventHandler
from .model import DTYPES, MODELS, load_model

_LOGGER = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wyoming Chatterbox TTS server")
    parser.add_argument(
        "--uri",
        required=True,
        help="Server URI (e.g., tcp://0.0.0.0:10201)",
    )
    parser.add_argument(
        "--voice-ref",
        required=True,
        help="Path to voice reference WAV file (10-30s of speech)",
    )
    parser.add_argument(
        "--volume-boost",
        type=float,
        default=3.0,
        help="Output volume multiplier (default: 3.0)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="standard",
        choices=MODELS,
        help="Chatterbox variant (default: standard)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=list(DTYPES),
        help="Precision for the T3 model (default: float32)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Validate voice reference
    voice_ref = Path(args.voice_ref)
    if not voice_ref.exists():
        _LOGGER.error("Voice reference file not found: %s", voice_ref)
        return 1

    asyncio.run(run_server(args, str(voice_ref)))
    return 0


async def run_server(args, voice_ref: str):
    """Run the Wyoming server."""
    _LOGGER.info(
        "Loading Chatterbox %s model on %s (T3 %s)...", args.model, args.device, args.dtype
    )
    model = load_model(args.model, args.device, args.dtype)

    # Embed the reference voice once instead of on every request
    _LOGGER.info("Preparing voice: %s", voice_ref)
    model.prepare_conditionals(voice_ref)
    _ = model.generate("Ready.")

    _LOGGER.info("Starting server at %s (volume boost: %.1fx)", args.uri, args.volume_boost)

    server = AsyncServer.from_uri(args.uri)
    await server.run(
        partial(
            ChatterboxEventHandler,
            model=model,
            generate_lock=asyncio.Lock(),
            sample_rate=model.sr,
            volume_boost=args.volume_boost,
        )
    )


if __name__ == "__main__":
    main()
