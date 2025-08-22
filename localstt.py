from functools import lru_cache
from pathlib import Path
from typing import Protocol

import click
import librosa
import numpy as np
from numpy.typing import NDArray

from faster_whisper import WhisperModel

curr_dir = Path(__file__).parent

class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...

class LocalFasterWhisperSTT(STTModel):
    def __init__(self, model_path: str = "small", device: str = "cpu", compute_type: str = "int8"):
        try:
            self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
        except Exception as e:
            raise RuntimeError(f"Error loading faster-whisper model: {e}")

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        """Transcribe audio to text using automatic language detection."""
        sr, audio_np = audio  # sr is 48000, audio_np is shape (1, N), dtype=int16

        # Convert int16 to float32 and normalize to [-1, 1]
        audio_np = audio_np.astype(np.float32) / 32768.0

        # Convert shape (1, N) to (N,) for mono
        audio_np = audio_np[0]

        # Resample from 48 kHz to 16 kHz
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

        # Transcribe with faster-whisper using automatic language detection
        segments, _ = self.model.transcribe(audio_np, beam_size=5, language=None)
        return " ".join(seg.text for seg in segments)

@lru_cache
def get_stt_model(
    model_path: str = "small", device: str = "cpu", compute_type: str = "int8"
) -> STTModel:
    os_env = __import__("os").environ
    os_env["TOKENIZERS_PARALLELISM"] = "false"

    m = LocalFasterWhisperSTT(model_path, device, compute_type)

    # Warm-up with 1 second of silence
    dummy_audio = np.zeros((1, 48000), dtype=np.int16) # raw audio
    print(click.style("INFO", fg="green") + ":\t  Warming up STT model.")
    m.stt((48000, dummy_audio))
    print(click.style("INFO", fg="green") + ":\t  STT model warmed up.")
    return m

# Assuming AudioChunk has 'start' and 'end' in frames
def stt_for_chunks(
    stt_model: STTModel,
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chunks: list[dict],  # Each dict: {"start": int, "end": int}
) -> str:
    sr, audio_np = audio
    return " ".join(
        stt_model.stt((sr, audio_np[chunk["start"] : chunk["end"]]))
        for chunk in chunks
    )