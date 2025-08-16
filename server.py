import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_tts_model,
    get_twilio_turn_credentials,
    AlgoOptions,
    SileroVadOptions,
)
from gradio.utils import get_space
from ollama import chat
from pydantic import BaseModel
from localstt import get_stt_model
import soundfile as sf

load_dotenv()
curr_dir = Path(__file__).parent

tts_model = get_tts_model()
stt_model = get_stt_model("faster-whisper-small.en", device="cpu")

def save_audio(audio: tuple[int, np.ndarray], filename="received_audio.wav"):
    sample_rate, audio_array = audio

    # Ensure float32 or int16
    if audio_array.dtype not in (np.float32, np.int16):
        audio_array = audio_array.astype(np.float32)

    # Ensure correct shape: (samples,) or (samples, channels)
    if audio_array.ndim == 1:
        pass  # already mono
    elif audio_array.ndim == 2:
        audio_array = audio_array.T if audio_array.shape[0] < audio_array.shape[1] else audio_array
    else:
        raise ValueError(f"Unexpected audio shape: {audio_array.shape}")

    # Save as WAV PCM16
    sf.write(filename, audio_array, sample_rate, subtype="PCM_16")

def _dtype_to_sf_str(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "float32"
    if dtype == np.int16:
        return "int16"
    # fallback
    return "float32"

def load_audio_from_file(
    filename: str,
    orig_audio: tuple[int, np.ndarray] | None = None,
    prefer_dtype: np.dtype | None = None,
    force_mono: bool = False,
    transpose_if_needed: bool = True,
    verbose: bool = False,
) -> tuple[int, np.ndarray]:
    """
    Load audio from `filename` and try to match the dtype and orientation of `orig_audio`.

    Parameters
    ----------
    filename :
        Path to saved WAV (or other readable audio) file.
    orig_audio :
        Optional original (sample_rate, array) tuple you previously saved. If provided,
        the returned array will be coerced to the same dtype / shape orientation when possible.
    prefer_dtype :
        If provided, prefer this dtype (np.float32 / np.int16). Overrides orig_audio dtype.
    force_mono :
        If True and result is multi-channel, collapse to mono by averaging channels.
    transpose_if_needed :
        If True, attempt to transpose reloaded array if that makes the shapes match orig_audio.
    verbose :
        Print debug info.

    Returns
    -------
    (sr, arr)
        sample rate and numpy array (shape either (samples,) or (samples, channels))
    """
    sr = None
    arr = None

    # decide target dtype string for soundfile (if available)
    target_dtype = None
    if prefer_dtype is not None:
        target_dtype = prefer_dtype
    elif orig_audio is not None:
        target_dtype = orig_audio[1].dtype

    sf_dtype = _dtype_to_sf_str(target_dtype) if target_dtype is not None else "float32"

    # 1) try soundfile (libsndfile) first
    try:
        import soundfile as sf

        arr, sr = sf.read(filename, dtype=sf_dtype)
        if verbose:
            print(f"[load_audio_from_file] soundfile read ok -> sr={sr}, arr.shape={arr.shape}, arr.dtype={arr.dtype}")
    except Exception as e:
        if verbose:
            print(f"[load_audio_from_file] soundfile failed ({e}), falling back to scipy.io.wavfile")
        # fallback: scipy.io.wavfile
        try:
            from scipy.io import wavfile

            sr, arr = wavfile.read(filename)
            arr = np.asarray(arr)
            if verbose:
                print(f"[load_audio_from_file] scipy read ok -> sr={sr}, arr.shape={arr.shape}, arr.dtype={arr.dtype}")
        except Exception as e2:
            raise RuntimeError(f"Failed to read audio file '{filename}': {e2}") from e2

    # Now arr is loaded. Normalize types and shapes:
    # If scipy returned int16/int32 but you wanted float32, convert appropriately.
    if prefer_dtype is not None:
        final_dtype = prefer_dtype
    elif orig_audio is not None:
        final_dtype = orig_audio[1].dtype
    else:
        final_dtype = arr.dtype  # keep whatever we got

    # If arr is int (e.g., int16) and user wants float32, convert to float in [-1,1]
    if arr.dtype.kind in ("i", "u") and final_dtype == np.float32:
        # assume 16-bit PCM if int16
        if arr.dtype == np.int16:
            arr = (arr.astype(np.float32) / 32767.0)
        else:
            # generic integer -> float conversion
            max_val = float(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / max_val
        if verbose:
            print(f"[load_audio_from_file] converted integer -> float32")
    elif arr.dtype.kind == "f" and final_dtype == np.int16:
        # convert float [-1,1] -> int16
        clipped = np.clip(arr, -1.0, 1.0)
        arr = (clipped * 32767.0).astype(np.int16)
        if verbose:
            print(f"[load_audio_from_file] converted float -> int16")
    else:
        # cast if needed
        if arr.dtype != final_dtype:
            try:
                arr = arr.astype(final_dtype)
                if verbose:
                    print(f"[load_audio_from_file] casted arr.dtype -> {final_dtype}")
            except Exception:
                # keep arr as-is if cast fails
                if verbose:
                    print("[load_audio_from_file] dtype cast failed, keeping loaded dtype")

    # Ensure shape is (samples,) or (samples, channels)
    if arr.ndim == 2:
        # common: soundfile returns (samples, channels)
        # some providers give (channels, samples) — try to detect and fix using orig_audio
        if orig_audio is not None:
            orig_arr = orig_audio[1]
            if orig_arr.ndim == 2 and transpose_if_needed:
                if arr.shape != orig_arr.shape and arr.T.shape == orig_arr.shape:
                    if verbose:
                        print("[load_audio_from_file] transposing loaded array to match original orientation")
                    arr = arr.T
            # if original had shape (channels, samples) and arr is (samples, channels) but sizes match when transposed
            if orig_arr.ndim == 2 and arr.shape != orig_arr.shape and arr.T.shape == orig_arr.shape:
                arr = arr.T
        # If user forced mono, average channels
        if force_mono:
            if arr.ndim == 2:
                arr = arr.mean(axis=1)
                if verbose:
                    print("[load_audio_from_file] collapsed to mono by averaging channels")
    elif arr.ndim == 1:
        # arr is mono. If original was 2D and lengths match, try to reshape.
        if orig_audio is not None:
            orig_arr = orig_audio[1]
            if orig_arr.ndim == 2 and arr.size == orig_arr.size:
                try:
                    arr = arr.reshape(orig_arr.shape)
                    if verbose:
                        print("[load_audio_from_file] reshaped mono -> original 2D shape")
                except Exception:
                    pass

    # final safety: if original dtype requested and we didn't match, try final cast
    if orig_audio is not None:
        target_final = orig_audio[1].dtype
        if arr.dtype != target_final:
            try:
                arr = arr.astype(target_final)
                if verbose:
                    print(f"[load_audio_from_file] final cast to original dtype {target_final}")
            except Exception:
                if verbose:
                    print("[load_audio_from_file] final cast failed; keeping current dtype")

    return int(sr), arr

def response(
    audio: tuple[int, np.ndarray],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]

    save_audio(audio, "received_audio.wav")
    audio_from_file = load_audio_from_file("received_audio.wav", orig_audio=audio, verbose=True)


    prompt = stt_model.stt(audio_from_file)
    if not prompt:
        return
    print(f'user>{prompt}')
    chatbot.append({"role": "user", "content": prompt})
    yield AdditionalOutputs(chatbot)
    messages.append({"role": "user", "content": prompt})

    resp = chat(model="llama3.2:1b", messages=messages)
    response_text = resp["message"]["content"]
    print(f'assistant>{response_text}')
    chatbot.append({"role": "assistant", "content": response_text})

    start = time.time()
    print("starting tts", start)
    for i, chunk in enumerate(tts_model.stream_tts_sync(response_text)):
        print("chunk", i, time.time() - start)
        yield chunk
    print("finished tts", time.time() - start)
    yield AdditionalOutputs(chatbot)

chatbot = gr.Chatbot(type="messages")
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(
        response,# Algorithm-level options (how you collect / decide on chunks)
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,        # collect 0.6s per internal chunk (bigger => more context)
            started_talking_threshold=0.2,   # fraction of the chunk that must be speech to mark "start"
            speech_threshold=0.1             # lower => more sensitive to soft speech
        ),
        # Model-level VAD (Silero) options (controls sensitivity / min durations)
        model_options=SileroVadOptions(
            threshold=0.5,                   # VAD decision threshold (lower => more sensitive)
            min_speech_duration_ms=250,      # minimum speech length to consider (short words allowed)
            min_silence_duration_ms=100      # silence required to consider speech ended
        ),
    ),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

class Message(BaseModel):
    role: str
    content: str

class InputData(BaseModel):
    webrtc_id: str
    chatbot: list[Message]

app = FastAPI()
stream.mount(app)

@app.get("/")
async def _():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (curr_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/input_hook")
async def _(body: InputData):
    stream.set_input(body.webrtc_id, body.model_dump()["chatbot"])
    return {"status": "ok"}

@app.get("/outputs")
def _(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            chatbot = output.args[0]
            yield f"event: output\ndata: {json.dumps(chatbot[-1])}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860, server_name="0.0.0.0")
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)