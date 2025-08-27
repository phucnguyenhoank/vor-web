# -----------SSL (Optional)--------------------
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
import ast
import os
import time
import re
from pathlib import Path
import numpy as np
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
from order_functions import *
from HumAwareVad.humaware_vad import HumAwareVADModel

available_tools = {
    "get_menu": get_menu,
    "increase_order_items": increase_order_items,
    "remove_order_items": remove_order_items,
    "set_order_items": set_order_items,
    "replace_order_item": replace_order_item,
    "preview_order": preview_order,
    "create_order": create_order
}

tool_list = [
    get_menu,
    increase_order_items,
    remove_order_items,
    set_order_items,
    replace_order_item,
    preview_order,
    create_order
]

system_prompt = f"""
You are a drive-thru staff member at a fast-food restaurant. Suggest foods, related items, and combos based on customer needs.
Your job is to talk naturally with customers using short, polite, friendly sentences. 
You always stay in character as a staff member.
You have to create an order to finalize it.

Your goals:
1. Greet the customer warmly and ask what they would like to order.
2. Collect the customer's order by calling the appropriate tools:
   - Call get_menu to answer questions about available food, drinks, and related combos.
   - Call increase_order_items when the customer adds food or combo of foods.
   - Call set_order_items when the customer specifies exact quantities.
   - Call replace_order_items if the customer requests replacing an existing item with a different one.
   - Call remove_order_items when the customer changes their mind or removes items.
   - Call create_order when the customer confirms the order is complete to finalize the order, even if the customer 
     does not explicitly say "create order". Instead, listen for natural 
     signals like: "Yes", "That's all", "I'm done", "That's it", or when the 
     customer confirms the order is complete.
   - Call preview_order only to check what's currently in the customer's cart or to show the price of their order so far.
3. Always confirm the final order with the customer before creating it. 
   Example: “Here's what I've got for you: [items]. Does that look correct?”
   Only after the customer confirms, then call create_order.
4. After create_order, tell the customer their total, thank them, and politely direct them to the next payment window.
5. Do NOT make up menu items, prices, or totals yourself — always rely on tools for facts.
6. Keep the conversation natural: confirm items, suggest combos if appropriate, but never overwhelm the customer.
7. If the customer seems done ordering, politely confirm: 
   “Is that everything for today?” and if yes, then review the order, confirm it with them, and then call create_order.
8. Stay brief, concise, and conversational. Avoid robotic long answers.

Refusing irrelevant questions:
- If the customer asks something unrelated to food, drink, or ordering, 
  politely decline and steer them back. Example: 
  “Sorry, I can only help you with your order today. Would you like to add something?”

Tool usage rules:
- Only call tools if needed to fulfill the user's request. 
- You can call many tools if you think that is needed.
- Always return to the conversation with the customer after a tool call. 
- Do NOT expose tool names or JSON arguments to the customer. 
  Just speak naturally as if you are a staff member.
- Example: if the model uses increase_order_items internally, 
  then to the customer you say: “Got it, I've added a cheeseburger to your order.”

End of interaction:
- After create_order is called and the order is confirmed, tell the customer their total, 
  thank them, and direct them to the next payment window. 
- End the conversation naturally.

Remember: you are the staff, not the system. Always be polite, clear, 
and helpful like a real fast-food employee at a drive-thru.
"""

greeting_prompt = "Welcome! What would you like to eat today?"

is_speaking = False
current_response = ""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "assistant", "content": greeting_prompt}
]
LLM_NAME =  "llama3.1:8b-instruct-q8_0" # "llama3.2:3b-instruct-fp16"

curr_dir = Path(__file__).parent

tts_model = get_tts_model()
stt_model = get_stt_model("faster-whisper-large-v3", device="auto")

def clean_for_tts(text: str) -> str:
    # Remove '*'
    cleaned = text.replace("*", "")

    def money_to_speech(match):
        amount = match.group(1)
        if "." in amount:
            dollars, cents = amount.split(".")
            if dollars == "0":  # like $0.99
                return f"{int(cents)} cents"
            if cents == "00":  # like $12.00
                return f"{int(dollars)} dollars"
            return f"{int(dollars)} {cents}"
        else:
            return f"{int(amount)} dollars"

    # Convert money
    cleaned = re.sub(r"\$([0-9]+(?:\.[0-9]+)?)", money_to_speech, cleaned)

    return cleaned

def print_content(message):
    role = message["role"]
    content = message["content"]
    print(f"{role}> {content}")

def safe_parse_arguments(arguments: dict) -> dict:
    """
    Ensure all values inside arguments are parsed into Python objects.
    Handles JSON strings, Python-style reprs, and nested structures.
    """
    def try_parse(value):
        if isinstance(value, str):
            # First try JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
            # Then try Python literal (handles single quotes)
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        elif isinstance(value, dict):
            return {k: try_parse(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [try_parse(v) for v in value]
        return value

    return {k: try_parse(v) for k, v in arguments.items()}

def startup():
    for chunk in tts_model.stream_tts_sync(greeting_prompt):
        yield chunk


def is_complete_sentence(text: str) -> bool:
    """Check if text is a complete sentence ending with '.', '?' or '!' but not part of a number."""
    text = text.strip()
    if not text.endswith((".", "?", "!")):
        return False
    # Reject if the period is part of a number (e.g., '1.49' or '1.')
    # A period is part of a number if preceded by a digit and followed by a digit or nothing
    if text.endswith("."):
        # Check the last few characters to ensure the period isn't part of a number
        # Use regex to find the last period and its context
        last_period_index = text.rfind(".")
        if last_period_index > 0:
            before_char = text[last_period_index - 1] if last_period_index > 0 else ""
            after_char = text[last_period_index + 1] if last_period_index + 1 < len(text) else ""
            # If the period is preceded by a digit and followed by a digit or nothing, it might be part of a number
            if before_char.isdigit():
                if after_char.isdigit() or not after_char:
                    # Check if the period is part of a valid number (e.g., '1.49' or '1.')
                    # Use regex to match the last number-like pattern before the period
                    number_pattern = re.search(r"\d+\.\d*$|\d+\.$", text)
                    if number_pattern and number_pattern.end() == len(text):
                        return False
    return True

def flush_buffer(buffer: str, buffer_list: list[str], tts_model, clean_for_tts):
    """Clean and convert a finished sentence to audio chunks, then add to buffer_list"""
    raw = buffer.strip()
    clean = raw.replace("*", "")
    print(f"Raw:   {raw}")
    print(f"Speak: {clean}")
    buffer_list.append(clean)
    for chunk in tts_model.stream_tts_sync(clean_for_tts(clean)):
        yield chunk
    return ""  # reset buffer

def response(audio: tuple[int, np.ndarray]):
    """Streamed response function for processing audio input and generating TTS output"""
    global is_speaking, current_response, messages
    print('Got it')
    
    # Transcribe audio using stt_model (handles 48 kHz, int16, shape=(1, N))
    prompt = stt_model.stt(audio)

    # Check noise
    word_count = len(prompt.strip().split())
    if word_count == 0:  # or is_speaking and word_count <= 2:
        print(f'ignored_noise: [{prompt}] ({word_count} words)')
        # Restart talking the current response instead of stopping
        if current_response:
            is_speaking = True
            for chunk in tts_model.stream_tts_sync(clean_for_tts(current_response)):
                yield chunk
            is_speaking = False
        return  # End after restart

    print(f'user>{prompt}')
    messages.append({"role": "user", "content": prompt})

    # Initialize streaming chat
    stream = chat(model=LLM_NAME, messages=messages, tools=tool_list, stream=True)

    buffer = ""
    buffer_list = []
    tool_calls = []
    assistant_message = {"role": "assistant", "content": ""}

    for chunk in stream:
        content = chunk.message.get("content", "")
        if content:
            buffer += content
            assistant_message["content"] += content
            if is_complete_sentence(buffer):
                buffer = yield from flush_buffer(buffer, buffer_list, tts_model, clean_for_tts)

        # Check for tool calls in the chunk
        if chunk.message.get("tool_calls"):
            tool_calls.extend(chunk.message["tool_calls"])

    # Flush any remaining buffer content
    if buffer.strip():
        buffer = yield from flush_buffer(buffer, buffer_list, tts_model, clean_for_tts)
        assistant_message["content"] = "".join(buffer_list)

    # Update current_response
    current_response = assistant_message["content"]

    # Handle tool calls if any
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            if function_name in available_tools:
                function_to_call = available_tools[function_name]
                arguments = tool_call["function"]["arguments"]
                parsed_args = safe_parse_arguments(arguments)
                tool_result = function_to_call(**parsed_args)
                print(f'Tool {function_name} result:\n{json.dumps(tool_result, indent=2)}')
            else:
                tool_result = {"error": f"Unknown function: {function_name}"}

            messages.append({
                "role": "tool",
                "name": function_name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        # Follow-up streaming chat for tool results
        followup_stream = chat(model=LLM_NAME, messages=messages, tools=tool_list, stream=True)
        buffer = ""
        buffer_list = []
        followup_message = {"role": "assistant", "content": ""}

        for chunk in followup_stream:
            content = chunk.message.get("content", "")
            if content:
                buffer += content
                followup_message["content"] += content
                if is_complete_sentence(buffer):
                    buffer = yield from flush_buffer(buffer, buffer_list, tts_model, clean_for_tts)

        # Flush any remaining buffer content
        if buffer.strip():
            buffer = yield from flush_buffer(buffer, buffer_list, tts_model, clean_for_tts)
            followup_message["content"] = "".join(buffer_list)

        print_content(followup_message)
        current_response = followup_message["content"]
        is_speaking = True
        messages.append(followup_message)
    else:
        print_content(assistant_message)
        is_speaking = True
        messages.append(assistant_message)

    is_speaking = False
    print("\n--- Response Finished ---")

THRESHOLD = 0.4
stream = Stream(
    modality="audio",
    mode="send-receive",
    handler=ReplyOnPause(
        response,
        model=HumAwareVADModel(threshold=0.4), # small = more sensitive to small sounds
        startup_fn=startup,
        algo_options=AlgoOptions(
            audio_chunk_duration=THRESHOLD,
            started_talking_threshold=THRESHOLD - 0.1,
            speech_threshold=0.1
        ),
        # model_options=SileroVadOptions(
        #     threshold=0.9,                   # VAD decision threshold (lower => more sensitive)
        #     min_speech_duration_ms=100,      # minimum speech length to consider (short words allowed)
        #     min_silence_duration_ms=250      # silence required to consider speech ended
        # ),
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

