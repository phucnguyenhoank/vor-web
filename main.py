import re

s = "Your total comes out to be $12. Please drive up to the next window to pay and collect your food. Have a great day!"

import re

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


print(clean_for_tts(s))