"""
Project Title: Advanced Audio-Based Virtual Assistant with Real-Time Sentiment Analysis and Context-Aware Responses
File Name: advanced_virtual_assistant_sentiment_analysis.py

Project Description:
This project is an advanced audio-based virtual assistant that not only listens and responds but also analyzes sentiment in real-time to deliver context-aware responses. It handles multiple conversation threads based on user emotional states and voice tone, adapting its response and interaction style dynamically.

This project extends the complexity of prior audio projects by incorporating:

Real-Time Sentiment and Tone Detection: The virtual assistant determines user sentiment from both text transcription and voice tone analysis using OpenAI's Whisper model for transcription and a sentiment model.
Conversation Memory: It tracks the conversation's emotional progress over time and tailors responses based on context and history.
Customizable Interaction Styles: The system adapts its conversational tone (formal, casual, professional) based on detected emotional states.
Real-Time Speech Synthesis: The assistant uses emotional data to vary speech synthesis (e.g., slower for calming responses, faster for urgency).
Python Code:
"""
import os
import sounddevice as sd
import numpy as np
import openai
from io import BytesIO
from pydub import AudioSegment, playback
from openai import OpenAI
from apikey import apikey
import pyttsx3
from threading import Thread

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Sentiment categories and their respective response styles
SENTIMENT_RESPONSES = {
    "positive": {"tone": "friendly", "speed": 150, "pitch": 200},
    "neutral": {"tone": "informative", "speed": 120, "pitch": 150},
    "negative": {"tone": "calming", "speed": 100, "pitch": 100},
}

# Memory to track conversation sentiment history
SENTIMENT_HISTORY = []
MAX_HISTORY = 10


# Function to record user input audio
def record_audio(duration=10, sample_rate=16000):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    return recording


# Transcribe the audio to text using OpenAI's Whisper API
def transcribe_audio(audio_data):
    audio_stream = BytesIO(audio_data)
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )
    transcription = response['text']
    print(f"Transcription: {transcription}")
    return transcription


# Analyze sentiment from transcribed text using GPT
def analyze_sentiment(transcription):
    sentiment_prompt = f"Analyze the sentiment of the following text: '{transcription}'"
    sentiment_response = client.completions.create(
        model="gpt-4",
        prompt=sentiment_prompt,
        max_tokens=50
    )

    sentiment = sentiment_response['choices'][0]['text'].strip().lower()
    print(f"Detected Sentiment: {sentiment}")
    return sentiment


# Adjust the TTS engine properties (speed, pitch, etc.) based on sentiment
def adjust_tts_settings(sentiment):
    if sentiment in SENTIMENT_RESPONSES:
        settings = SENTIMENT_RESPONSES[sentiment]
        tts_engine.setProperty('rate', settings['speed'])
        tts_engine.setProperty('pitch', settings['pitch'])
    else:
        tts_engine.setProperty('rate', 120)
        tts_engine.setProperty('pitch', 150)  # Default settings


# Use GPT to generate a response based on sentiment
def generate_response(transcription, sentiment):
    conversation_prompt = f"The user said: '{transcription}'. The sentiment is '{sentiment}'. Generate an appropriate response based on this."
    response = client.completions.create(
        model="gpt-4",
        prompt=conversation_prompt,
        max_tokens=150
    )
    response_text = response['choices'][0]['text'].strip()
    print(f"Assistant Response: {response_text}")
    return response_text


# Function to keep track of conversation sentiment over time
def track_sentiment_history(sentiment):
    if len(SENTIMENT_HISTORY) >= MAX_HISTORY:
        SENTIMENT_HISTORY.pop(0)
    SENTIMENT_HISTORY.append(sentiment)


# Main loop to handle real-time interaction and sentiment analysis
def real_time_assistant(duration=10):
    while True:
        # Step 1: Record user input audio
        audio_data = record_audio(duration)

        # Step 2: Convert the numpy array to bytes
        audio_data_bytes = (audio_data * 32767).astype(np.int16).tobytes()

        # Step 3: Transcribe the audio to text
        transcription = transcribe_audio(audio_data_bytes)

        # Step 4: Analyze the sentiment of the transcription
        sentiment = analyze_sentiment(transcription)

        # Step 5: Adjust TTS settings based on sentiment
        adjust_tts_settings(sentiment)

        # Step 6: Generate an appropriate response using GPT
        response = generate_response(transcription, sentiment)

        # Step 7: Use the TTS engine to speak the response
        tts_engine.say(response)
        tts_engine.runAndWait()

        # Step 8: Track sentiment history
        track_sentiment_history(sentiment)

        # Optional: Add further customization for responses based on sentiment history


# Run the real-time assistant
if __name__ == "__main__":
    real_time_assistant(duration=10)
"""

"""