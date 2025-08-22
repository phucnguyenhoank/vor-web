from fastrtc.text_to_speech.tts import TTSModel, TTSOptions
import asyncio
import re
from typing import AsyncGenerator, Generator
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import torch
from transformers import VitsModel, AutoTokenizer

@dataclass
class VitsTTSOptions(TTSOptions):
    voice: str | None = None
    speed: float | None = None
    lang: str | None = None

class VitsTTSModel(TTSModel):
    def __init__(self):
        """Initialize VITS TTS model with the MMS-TTS Vietnamese model."""
        try:
            self.model = VitsModel.from_pretrained("facebook/mms-tts-vie")
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
        except Exception as e:
            raise RuntimeError(f"Failed to load VITS model or tokenizer: {str(e)}")

    def tts(
        self, text: str, options: VitsTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        """Generate speech waveform from text."""
        options = options or VitsTTSOptions()
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            # MMS-TTS uses 'waveform' for output
            waveform = outputs.waveform[0].cpu().numpy().astype(np.float32)
            return self.model.config.sampling_rate, waveform
        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {str(e)}")

    async def stream_tts(
        self, text: str, options: VitsTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        """Stream speech waveform by processing text sentence by sentence."""
        options = options or VitsTTSOptions()
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

        for s_idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            chunk_idx = 0
            _, waveform = self.tts(sentence, options)
            if s_idx != 0 and chunk_idx == 0:
                yield self.model.config.sampling_rate, np.zeros(
                    self.model.config.sampling_rate // 7, dtype=np.float32
                )
            chunk_idx += 1
            yield self.model.config.sampling_rate, waveform

    def stream_tts_sync(
        self, text: str, options: VitsTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        """Synchronous streaming by running async generator in a new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            iterator = self.stream_tts(text, options).__aiter__()
            while True:
                try:
                    yield loop.run_until_complete(iterator.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()