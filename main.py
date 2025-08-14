# server.py
import json
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# ---- your imports / models ----
from fastrtc import ReplyOnPause, Stream, get_tts_model, get_stt_model, AdditionalOutputs
from ollama import chat

# Use your preferred STT/TTS initialization
stt_model = get_stt_model()
tts_model = get_tts_model()

system_prompt = """
    You are a friendly, efficient fast-food restaurant receptionist who talks with customers through voice.
    Do NOT use asterisk, don't talk about any thing unrelated to your job.
    You manage customer carts and create orders.
    You always inform to customers what you have done with their cart.
    When speaking to customers, never mention item IDs, use item names instead.
    Ask clarifying questions when needed (e.g., size, toppings).
    Always preview the cart before create an order.
"""

greeting_prompt = "Welcome! I'm here to take your order — what would you like to eat today?"

# single global conversation (simple example)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "assistant", "content": greeting_prompt},
]

# generator used by ReplyOnPause
def echo(audio):
    """
    audio: provided by fastrtc (usually tuple describing the audio chunk)
    STT -> LLM -> yield AdditionalOutputs(messages) -> stream TTS audio
    """
    # transcribe (your stt_model has .stt(audio) in your snippet)
    print('hi')
    transcript = stt_model.stt(audio)
    print("transcript-raw:", transcript)

    if transcript:
        # append user text and notify UI
        messages.append({"role": "user", "content": transcript})
        # send user message to UI immediately
        yield AdditionalOutputs(messages)

        # call LLM (ollama.chat usage from your snippet)
        response = chat(model="llama3.2:1b", messages=messages)
        # ollama returns nested dict; this matches your earlier usage
        response_text = response["message"]["content"]
        messages.append({"role": "assistant", "content": response_text})

        # notify UI of assistant message before audio streaming
        yield AdditionalOutputs(messages)

        # stream TTS audio chunks back to client
        for chunk in tts_model.stream_tts_sync(response_text):
            yield chunk

# optional startup greeting generator (streams TTS once when stream created - optional)
def startup_greeting():
    for chunk in tts_model.stream_tts_sync("Welcome, thanks for choosing us, what would you like to eat today?"):
        yield chunk

# create fastrtc Stream with ReplyOnPause(echo)
stream = Stream(
    handler=ReplyOnPause(echo),
    modality="audio",
    mode="send-receive",
)

# --- mount to FastAPI and add endpoints ---
app = FastAPI()
stream.mount(app)  # registers the /webrtc/offer endpoint etc.

curr_dir = Path(__file__).parent

@app.get("/", response_class=HTMLResponse)
async def index():
    # serve index.html with optional RTC config replaced
    html = (curr_dir / "index.html").read_text()
    rtc_config = None  # set to get_twilio_turn_credentials() if needed
    html = html.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html, status_code=200)

# SSE endpoint the client listens to for textual outputs (messages)
@app.get("/outputs")
def outputs(webrtc_id: str):
    async def event_stream():
        # stream.output_stream yields output objects that wrap AdditionalOutputs
        async for output in stream.output_stream(webrtc_id):
            # output.args[0] should be the messages list we passed via AdditionalOutputs
            payload = output.args[0] if output.args else {}
            # send last message to the client as SSE
            # we'll send the entire messages list for simplicity
            yield f"event: output\ndata: {json.dumps(payload)}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# small helper to allow sending text from client (optional)
@app.post("/input_hook")
async def input_hook(req: Request):
    """
    Accepts JSON: {"webrtc_id": "<id>", "content": "<text>"}
    Appends as a user message and triggers stream.set_input so the stream sees the new input.
    """
    body = await req.json()
    webrtc_id = body.get("webrtc_id")
    content = body.get("content", "").strip()
    if not webrtc_id or not content:
        return {"status": "error", "reason": "missing webrtc_id or content"}
    messages.append({"role": "user", "content": content})
    # set input for that stream if you want to trigger anything server-side
    try:
        stream.set_input(webrtc_id, messages)
    except Exception:
        # if webrtc_id unknown, ignore
        pass
    return {"status": "ok"}


