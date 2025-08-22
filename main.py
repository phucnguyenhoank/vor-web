import re

s = "Xin chào! Thanks for choosing us — what would you like to eat today? Would you like to see our menu to decide?"

def detect_language(text: str) -> str:
    """Decide if a string is English or Vietnamese."""
    # Vietnamese-specific characters
    vietnamese_chars = "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ"
    
    if any(ch in text.lower() for ch in vietnamese_chars):
        return "vi"
    
    # Common Vietnamese words (expandable list)
    vietnamese_words = {"và", "của", "nhưng", "không", "tôi", "bạn", "chúng", "đây", "kia"}
    words = set(re.findall(r"\w+", text.lower()))
    if words & vietnamese_words:
        return "vi"
    
    # Fallback → assume English
    return "en"



print(detect_language(s) == "en")