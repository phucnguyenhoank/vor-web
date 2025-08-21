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

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastrtc import (
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
from localstt import get_stt_model

load_dotenv()
curr_dir = Path(__file__).parent

tts_model = get_tts_model()
stt_model = get_stt_model("faster-whisper-large-v3", device="auto") # optional vietnamese

import re  # Add this import for regex cleaning

def clean_for_tts(text: str) -> str:
    # Remove leading '*' or 'number.' followed by space
    cleaned = text.replace("*", "")
    return cleaned

is_speaking = False
current_response = ""
messages = []

def say(text):
    global is_speaking
    is_speaking = True
    for i, chunk in enumerate(tts_model.stream_tts_sync(clean_for_tts(text))):
        yield chunk
    is_speaking = False


def response(
    audio: tuple[int, np.ndarray],
):
    print('got it')
    global is_speaking, current_response, messages
    # Transcribe audio using stt_model (handles 48 kHz, int16, shape=(1, N))
    prompt = stt_model.stt(audio)

    # Check noise
    word_count = len(prompt.strip().split())
    if word_count == 0 or is_speaking and word_count <= 2:
        print(f'ignored_noise>{prompt} ({word_count} words)')
        # Restart talking the current response instead of stopping
        if current_response:
            response_text = current_response
            say(response_text)
        return  # End after restart

    print(f'user>{prompt}')
    messages.append({"role": "user", "content": prompt})

    resp = chat(model="llama3.2:1b", messages=messages)

    response_text = resp["message"]["content"]
    print(f"assistant>{response_text}")
    messages.append({"role": "assistant", "content": response_text})
    current_response = response_text  # Cache for potential restarts

    say(response_text)
    # is_speaking = True
    # for i, chunk in enumerate(tts_model.stream_tts_sync(clean_for_tts(response_text))):
    #     yield chunk
    # is_speaking = False

stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(
        response,# Algorithm-level options (how you collect / decide on chunks)
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,        # collect 0.6s per internal chunk (bigger => more context)
            started_talking_threshold=0.57,   #  < 0.6 fraction of the chunk that must be speech to mark "start"
            speech_threshold=0.1             # lower => more sensitive to soft speech
        ),
        # Model-level VAD (Silero) options (controls sensitivity / min durations)
        model_options=SileroVadOptions(
            threshold=0.55,                   # VAD decision threshold (lower => more sensitive)
            min_speech_duration_ms=100,      # minimum speech length to consider (short words allowed)
            min_silence_duration_ms=3000      # silence required to consider speech ended
        ),
    ),
    rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None
)

app = FastAPI()
stream.mount(app)

@app.get("/")
async def _():
    rtc_config = get_twilio_turn_credentials() if get_space() else None
    html_content = (curr_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)