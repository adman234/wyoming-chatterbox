"""Wyoming event handler for Chatterbox TTS."""

import asyncio
import logging
import re

import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Attribution, Info, TtsProgram, TtsVoice, Describe
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
MIN_SENTENCE_CHARS = 20
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, merging short fragments into the next one."""
    sentences, pending = [], ""
    for piece in _SENTENCE_END.split(text.strip()):
        pending = f"{pending} {piece}".strip()
        if len(pending) >= MIN_SENTENCE_CHARS:
            sentences.append(pending)
            pending = ""
    if pending:
        if sentences:
            sentences[-1] = f"{sentences[-1]} {pending}"
        else:
            sentences.append(pending)
    return sentences


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

            await self.write_event(
                AudioStart(
                    rate=self.sample_rate, width=SAMPLE_WIDTH, channels=CHANNELS
                ).event()
            )

            # Generate one sentence at a time in an executor: peak VRAM stays at
            # the size of the longest sentence and the first audio arrives sooner.
            # The model shares voice state between calls, so only one request
            # generates at a time.
            loop = asyncio.get_running_loop()
            async with self.generate_lock:
                for sentence in split_sentences(text):
                    wav_tensor = await loop.run_in_executor(None, self.model.generate, sentence)
                    await self._write_audio(wav_tensor)
                if torch.cuda.is_available():
                    # hand activation and kv cache memory back between requests
                    torch.cuda.empty_cache()

            await self.write_event(AudioStop().event())
            _LOGGER.info("Synthesis complete")
            return True

        return True

    async def _write_audio(self, wav_tensor: torch.Tensor) -> None:
        """Send generated audio as int16 PCM chunks."""
        wav_tensor = wav_tensor.cpu().squeeze()
        if wav_tensor.dim() == 0:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Apply volume boost and clamp
        wav_tensor = torch.clamp(wav_tensor * self.volume_boost, -1.0, 1.0)
        audio_data = (wav_tensor * 32767).to(torch.int16).numpy().tobytes()

        # Send audio in chunks (100ms each)
        chunk_size = self.sample_rate * SAMPLE_WIDTH * CHANNELS // 10
        for i in range(0, len(audio_data), chunk_size):
            await self.write_event(
                AudioChunk(
                    audio=audio_data[i : i + chunk_size],
                    rate=self.sample_rate,
                    width=SAMPLE_WIDTH,
                    channels=CHANNELS,
                ).event()
            )
