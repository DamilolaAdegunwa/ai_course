import os
from transformers import pipeline, logging
import certifi

# Enable detailed logging
logging.set_verbosity(logging.WARNING)


# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['ENV'] = 'dev'
os.environ['ENVIRONMENT'] = 'dev'

# Use Whisper for Speech-to-Text
whisper_pipeline = pipeline(model="openai/whisper-small")

# Example audio file
audio_file = r"C:\Users\damil\PycharmProjects\ai_course\resources\Erwin Smith's Words __ My Soldiers.mp4"

# Transcribe audio
transcription = whisper_pipeline(audio_file)
print(f"Transcription: {transcription['text']}")
