# -----------SSL--------------------
import ssl
import urllib3
import requests.sessions
import requests
# --- Disable SSL Verification ---
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
old_request = requests.sessions.Session.request
def unsafe_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)
requests.sessions.Session.request = unsafe_request
 
from requests.sessions import Session as OriginalSession
class UnsafeSession(OriginalSession):
    def request(self, *args, **kwargs):
        kwargs['verify'] = False
        return super().request(*args, **kwargs)
requests.Session = UnsafeSession

# ------------SSL--------------
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
    get_stt_model,
    get_twilio_turn_credentials,
    AlgoOptions,
    SileroVadOptions,
)
from gradio.utils import get_space
from ollama import chat
from pydantic import BaseModel
from localstt import get_stt_model
import soundfile as sf
import librosa

load_dotenv()
curr_dir = Path(__file__).parent

tts_model = get_tts_model()
stt_model = get_stt_model("faster-whisper-small.en", device="cpu")

def save_audio(audio: tuple[int, np.ndarray], filename="received_audio.wav", target_sr=16000):
    sample_rate, audio_array = audio

    # Convert to float32 and normalize if int16
    if audio_array.dtype == np.int16:
        audio_array = audio_array.astype(np.float32) / 32768.0
    elif audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Resample to 16 kHz if needed
    if sample_rate != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    # Ensure correct shape: (samples,) or (samples, channels)
    if audio_array.ndim == 1:
        pass  # already mono
    elif audio_array.ndim == 2:
        if audio_array.shape[0] < audio_array.shape[1]:
            audio_array = audio_array.T  # Transpose to (samples, channels)
        # Mix to mono if multi-channel
        if audio_array.shape[1] > 1:
            audio_array = np.mean(audio_array, axis=1)
    else:
        raise ValueError(f"Unexpected audio shape: {audio_array.shape}")

    # Save as WAV PCM16
    sf.write(filename, audio_array, sample_rate, subtype="PCM_16")

def response(
    audio: tuple[int, np.ndarray],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]

    # Examine the raw audio
    print(audio)

    prompt = stt_model.stt(audio)
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
    total_time = round(time.time() - start, 2)
    print(f"finished tts: {total_time} sec total")
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
            min_silence_duration_ms=500      # silence required to consider speech ended
        ),
    ),
    additional_outputs_handler=lambda a, b: b,
    additional_inputs=[chatbot],
    additional_outputs=[chatbot],
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None
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