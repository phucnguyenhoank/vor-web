import re
import torch
import soundfile as sf
from transformers import VitsModel, AutoTokenizer


def vclean_for_tts(text: str) -> str:
    # Remove '*'
    cleaned = text.replace("*", "")

    # --- Normalize numbers with comma decimal separator ---
    cleaned = re.sub(r"(\d),(\d+)", r"\1.\2", cleaned)

    # --- Handle money ---
    def money_to_speech(match):
        amount = match.group(1)
        if "." in amount:
            dollars, cents = amount.split(".")
            dollars = int(dollars)
            cents = int(cents)
            if dollars == 0:
                return f"{cents} cen"
            if cents == 0:
                return f"{dollars} đô"
            return f"{dollars} đô {cents} cen"
        else:
            return f"{int(amount)} đô"

    cleaned = re.sub(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:\$|USD|đô la|đô)",
        money_to_speech,
        cleaned,
        flags=re.IGNORECASE
    )

    # --- English → Vietnamese food mapping ---
    FOOD_TRANSLATION = {
        "Burgers": "Bánh bơ gơ",
        "Combo": "Côm bô",
        "Cheeseburger": "Bánh bơ gơ phô mai",
        "Veggie Burger": "Bánh bơ gơ rau củ",
        "French Fries": "Khoai tây chiên",
        "Coca-Cola": "Cô-ca",
        "Orange Juice": "Nước cam",
        "Spicy Chicken Burger": "Bánh bơ gơ gà cay",
    }

    for en, vi in FOOD_TRANSLATION.items():
        cleaned = re.sub(rf"\b{en}\b", vi, cleaned, flags=re.IGNORECASE)

    return cleaned


# --- Input text ---
s = """Chào bạn! Hôm nay chúng tôi có các món ăn sau: 

- Món chính: Cheeseburger, Veggie Burger
- Sides: French Fries
- Trà sữa: Coca-Cola, Orange Juice

Bạn muốn chọn món gì hôm nay?"""

# Clean text
o = vclean_for_tts(s)

# --- Load model ---
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

inputs = tokenizer(o, return_tensors="pt")

# Generate speech
with torch.no_grad():
    output = model(**inputs).waveform

# Save to file (16kHz sample rate)
sf.write("hiiiii.wav", output.squeeze().cpu().numpy(), 16000)

print("✅ Xuất file thành công")
