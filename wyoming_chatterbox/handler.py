"""Wyoming event handler for Chatterbox TTS."""

import asyncio
import logging

import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, Describe
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)


class ChatterboxEventHandler(AsyncEventHandler):
    """Event handler for Chatterbox TTS."""

    def __init__(
        self,
        reader,
        writer,
        model,
        generate_lock: asyncio.Lock,
        sample_rate: int = 24000,
        volume_boost: float = 3.0,
    ):
        super().__init__(reader, writer)
        self.model = model
        self.generate_lock = generate_lock
        self.sample_rate = sample_rate
        self.volume_boost = volume_boost

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events."""
        if Describe.is_type(event.type):
            info = Info(
                tts=[
                    TtsProgram(
                        name="chatterbox",
                        description="Chatterbox TTS with voice cloning",
                        attribution=Attribution(
                            name="Resemble AI",
                            url="https://github.com/resemble-ai/chatterbox",
                        ),
                        installed=True,
                        version="1.0.0",
                        voices=[
                            TtsVoice(
                                name="custom",
                                description="Custom cloned voice",
                                attribution=Attribution(name="Custom", url=""),
                                installed=True,
                                version="1.0.0",
                                languages=["en"],
                            )
                        ],
                    )
                ]
            )
            await self.write_event(info.event())
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            text = synthesize.text
            _LOGGER.info("Synthesizing: %s", text)

            # Generate audio in executor to avoid blocking. The model shares
            # voice state between calls, so only one generation runs at a time.
            loop = asyncio.get_running_loop()
            async with self.generate_lock:
                wav_tensor = await loop.run_in_executor(None, self.model.generate, text)
                if torch.cuda.is_available():
                    # hand activation and kv cache memory back between requests
                    torch.cuda.empty_cache()

            # Convert to int16 PCM
            wav_tensor = wav_tensor.cpu().squeeze()
            if wav_tensor.dim() == 0:
                wav_tensor = wav_tensor.unsqueeze(0)

            # Apply volume boost and clamp
            wav_tensor = wav_tensor * self.volume_boost
            wav_tensor = torch.clamp(wav_tensor, -1.0, 1.0)
            wav_int16 = (wav_tensor * 32767).to(torch.int16)
            audio_data = wav_int16.numpy().tobytes()

            sample_rate = self.sample_rate
            sample_width = 2  # 16-bit
            channels = 1

            # Send audio start
            await self.write_event(
                AudioStart(
                    rate=sample_rate, width=sample_width, channels=channels
                ).event()
            )

            # Send audio in chunks (100ms each)
            chunk_size = sample_rate * sample_width * channels // 10
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await self.write_event(
                    AudioChunk(
                        audio=chunk,
                        rate=sample_rate,
                        width=sample_width,
                        channels=channels,
                    ).event()
                )

            await self.write_event(AudioStop().event())
            _LOGGER.info("Synthesis complete")
            return True

        return True
