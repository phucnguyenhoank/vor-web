from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import scipy

# Load model + tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

text = "Xin chào, tôi rất vui được gặp bạn. Hôm nay bạn có khỏe không?"
inputs = tokenizer(text, return_tensors="")

with torch.no_grad():
    output = model(**inputs).waveform

print("=== Raw model output (Torch Tensor) ===")
print("Shape:", output.shape)
print("Dtype:", output.dtype)
print("Sample values:", output[0, :10])  # first 10 samples

# 1. Torch → NumPy
if isinstance(output, torch.Tensor):
    output = output.cpu().numpy()

print("\n=== After .numpy() ===")
print("Shape:", output.shape)
print("Dtype:", output.dtype)
print("Sample values:", output[0, :10])

# 2. Remove batch dimension if present
if output.ndim > 1 and output.shape[0] == 1:
    output = output.squeeze(0)

print("\n=== After .squeeze(0) ===")
print("Shape:", output.shape)
print("Dtype:", output.dtype)
print("Sample values:", output[:10])

# 3. Convert float32 → int16
if output.dtype.kind == "f":
    output = (output * 32767).astype("int16")

print("\n=== After scaling to int16 ===")
print("Shape:", output.shape)
print("Dtype:", output.dtype)
print("Sample values:", output[:10])

# 4. Save WAV
scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output)
print("\nSaved techno.wav at sample rate", model.config.sampling_rate)
