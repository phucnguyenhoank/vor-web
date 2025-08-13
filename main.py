from fastrtc import ReplyOnPause, Stream, get_tts_model, get_stt_model
from ollama import chat
# from distil_whisper_fastrtc import get_stt_model

  # distil-whisper/distil-small.en (default), distil-medium.en, distil-large-v2, distil-large-v3

"""
from distil_whisper_fastrtc import DistilWhisperSTT

# Configure with specific device and precision
stt_model = DistilWhisperSTT(
    model="distil-whisper/distil-medium.en",
    device="cuda",  # Use GPU if available
    dtype="float16"  # Use half precision for faster inference
)
"""

stt_model = get_stt_model()
tts_model = get_tts_model()  # kokoro


def echo(audio):
    transcript = stt_model.stt(audio)
    print(f'transcript-raw:{transcript}')
    if transcript:
        # print(f'transcript:{transcript}')
        response = chat(
            model="llama3.2:1b", messages=[{"role": "user", "content": transcript}]
        )
        response_text = response["message"]["content"]
        print(f'response:{response_text}')
        for audio_chunk in tts_model.stream_tts_sync(response_text):
            yield audio_chunk


stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    stream.ui.launch(share=True)
