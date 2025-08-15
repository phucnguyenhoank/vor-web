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
    get_twilio_turn_credentials,
    get_stt_model,
    AlgoOptions, 
    SileroVadOptions,
)
from fastrtc.utils import audio_to_bytes
from gradio.utils import get_space
from ollama import chat
from pydantic import BaseModel
from distil_whisper_fastrtc import DistilWhisperSTT # , get_stt_model

load_dotenv()
curr_dir = Path(__file__).parent

tts_model = get_tts_model(device="cuda", dtype="float16")

stt_model = get_stt_model()

def startup():
    for chunk in tts_model.stream_tts_sync("Welcome! Thanks for choosing us. What would you like to eat today!"):
        yield chunk

def response(
    audio: tuple[int, np.ndarray],
    chatbot: list[dict] | None = None,
):
    chatbot = chatbot or []
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]

    prompt = stt_model.stt(audio)
    print(f'user>:{prompt}')
    if not prompt:
        return

    chatbot.append({"role": "user", "content": prompt})
    yield AdditionalOutputs(chatbot)
    messages.append({"role": "user", "content": prompt})

    resp = chat(model="llama3.2:1b", messages=messages)
    # Nanoseconds
    total_ns = resp.get("total_duration")
    eval_ns = resp.get("eval_duration")

    # Convert to seconds
    total_s = total_ns / 1e9 if total_ns is not None else None
    eval_s = eval_ns / 1e9 if eval_ns is not None else None
    
    print(f"Inference (eval) duration: {eval_s:.3f} s")
    print(f"Total duration: {total_s:.3f} s")

    response_text = resp["message"]["content"]
    print(f'assistant>:{response_text}')
    chatbot.append({"role": "assistant", "content": response_text})

    start = time.time()
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
    handler=ReplyOnPause(response,
        startup_fn=startup,
        algo_options=AlgoOptions(
            audio_chunk_duration=0.6,        # process 0.6s audio chunks
            started_talking_threshold=0.35,   # require louder start to detect speech
            speech_threshold=0.1              # overall speech confidence threshold
        ),
        model_options=SileroVadOptions(
            threshold=0.65,                   # VAD activation threshold <= .65
            min_speech_duration_ms=500,      # ignore speech shorter than 250ms
            min_silence_duration_ms=500      # need at least 500ms silence to stop
        )),
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