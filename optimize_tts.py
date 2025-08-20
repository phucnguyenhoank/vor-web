import ollama
import re

# ---- Helpers ----
def add_numbers(a: int, b: int) -> int:
    """Example tool: adds two numbers"""
    return a + b

def is_complete_sentence(text: str) -> bool:
    """Check if text is a complete sentence ending with '.' but not just a number like '1.'"""
    text = text.strip()
    if not text.endswith((".", "?", "!")):
        return False
    if re.fullmatch(r"\d+\.", text):  # reject "1.", "23." alone
        return False
    return True

def flush_buffer(buffer: str, buffer_list: list[str]):
    """Clean and print a finished sentence, then add to buffer_list"""
    raw = buffer.strip()
    clean = raw.replace("*", "")
    print(f"Raw:   {raw}")
    print(f"Speak: {clean}")
    buffer_list.append(clean)
    return ""  # reset buffer


# ---- Tool definition (not used here, but ready for later) ----
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers together",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# ---- Chat stream ----
stream = ollama.chat(
    model="llama3.2:1b",
    messages=[{"role": "user", "content": "Why is the sky blue"}],
    stream=True
)

buffer = ""
buffer_list = []

for chunk in stream:
    content = chunk.message.get("content", "")
    if not content:
        continue

    buffer += content
    if is_complete_sentence(buffer):
        buffer = flush_buffer(buffer, buffer_list)

# flush leftovers at the end
if buffer.strip():
    buffer = flush_buffer(buffer, buffer_list)

print("\n--- Conversation Finished ---")
