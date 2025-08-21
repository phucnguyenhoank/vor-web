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
import re
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
from function_descriptions import *
from order_funtions import *

available_tools = {
    "get_menu": get_menu,
    "get_customer_order": get_customer_order,
    "increase_order_items": increase_order_items,
    "remove_order_items": remove_order_items,
    "set_order_items": set_order_items,
    "calculate_total": calculate_total,
    "create_order": create_order
}

system_prompt = f"""
    You are a drive-thru staff member at a fast-food restaurant. 
    Your job is to talk naturally with customers using short, polite, friendly sentences. 
    You always stay in character as a staff member. 

    Your goals:
    1. Greet the customer warmly and ask what they would like to order.
    2. Collect the customer’s order by calling the appropriate tools:
    - Use get_menu to answer questions about available food and drinks.
    - Use increase_order_items when the customer adds something.
    - Use set_order_items when the customer specifies exact quantities.
    - Use remove_order_items when the customer changes their mind or removes items.
    - Use get_customer_order to confirm what’s currently in the cart.
    - Use calculate_total if the customer asks for the price so far.
    - Use create_order only when the order is finalized.
    3. After create_order, tell the customer their total and politely direct them to the next payment window.
    4. Do NOT make up menu items, prices, or totals yourself — always rely on tools for facts.
    5. Keep the conversation natural: confirm items, suggest combos if appropriate, but never overwhelm the customer.
    6. If the customer seems done ordering, ask politely if the order is complete. 
    Only then finalize with create_order.
    7. Stay brief and conversational. Avoid robotic long answers.

    Tool usage rules:
    - Only call a tool if needed to fulfill the user’s request. 
    - Always return to the conversation with the customer after a tool call. 
    - Do not expose tool names or JSON arguments to the customer. 
    Just speak naturally as if you are a staff member.
    - Example: if the model uses increase_order_items internally, 
    then to the customer you say: “Got it, I’ve added a cheeseburger to your order.”

    End of interaction:
    - After create_order is called, tell the customer their total, 
    thank them, and direct them to the next payment window. 
    - End the conversation naturally.

    Remember: you are the staff, not the system. Always be polite, clear, 
    and helpful like a real fast-food employee at a drive-thru.
    """

greeting_prompt = "Welcome! I'm here to take your order — what would you like to eat today?"


load_dotenv()
curr_dir = Path(__file__).parent

tts_model = get_tts_model()
stt_model = get_stt_model("faster-whisper-large-v3", device="auto") # optional vietnamese


def clean_for_tts(text: str) -> str:
    # Remove leading '*' or 'number.' followed by space
    cleaned = text.replace("*", "")
    return cleaned

def say(text):
    global is_speaking
    is_speaking = True
    for i, chunk in enumerate(tts_model.stream_tts_sync(clean_for_tts(text))):
        yield chunk
    is_speaking = False

is_speaking = False
current_response = ""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "assistant", "content": greeting_prompt}
]
LLM_NAME = "llama3.1:8b" # "llama3.2:1b"

def print_content(message):
    role = message["role"]
    content = message["content"]
    print(f"{role}> {content}")

def response(
    audio: tuple[int, np.ndarray],
):
    print('Got it')
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

    resp = chat(model=LLM_NAME, messages=messages, tools=TOOLS)

    message = resp["message"]
    print(message)
    if message.get("tool_calls"):
        tool_calls = message["tool_calls"]
        messages.append(message) # Append assistant's message with tool calls

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            if function_name in available_tools:
                function_to_call = available_tools[function_name]
                arguments = tool_call["function"].get("arguments", {})
                tool_result = function_to_call(**arguments)
            else:
                tool_result = {"error": f"Unknown function: {function_name}"}

            messages.append({
                "role": "tool",
                "name": function_name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        followup = chat(model=LLM_NAME, messages=messages, tools=TOOLS)
        print_content(followup["message"])
        current_response = followup["message"]["content"]
        is_speaking = True
        for i, chunk in enumerate(tts_model.stream_tts_sync(clean_for_tts(current_response))):
            yield chunk
        is_speaking = False
        messages.append(followup["message"])
    else:
        print_content(message)
        current_response = message["content"]
        is_speaking = True
        for i, chunk in enumerate(tts_model.stream_tts_sync(clean_for_tts(current_response))):
            yield chunk
        is_speaking = False
        messages.append(message)
        

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