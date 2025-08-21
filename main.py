import re

s = "Your total comes out to be $1.59. Please drive up to the next window to pay and collect your food. Have a great day!"

import re

def clean_for_tts(text: str) -> str:
    # Remove '*'
    cleaned = text.replace("*", "")

    # Convert $12 or $12.50 → "12 dollars" or "12.50 dollars"
    cleaned = re.sub(r"\$([0-9]+(?:\.[0-9]+)?)", r"\1 dollars", cleaned)

    return cleaned


print(clean_for_tts(s))