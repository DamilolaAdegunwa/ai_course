import os
import random
import time
from openai import OpenAI
from apikey import apikey  # Assuming your API key is stored in apikey.py
import speech_recognition as sr
import pyttsx3
import numpy as np

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
client = OpenAI()

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Background soundscapes
background_sounds = {
    "calm": "calm_background.wav",
    "nature": "nature_sounds.wav",
    "ocean": "ocean_waves.wav",
}


def analyze_speech(audio_data):
    # Simulated emotion detection (replace with actual emotion detection logic)
    pitch = np.random.choice(["high", "low", "neutral"])  # Simulated pitch detection
    emotion = np.random.choice(["happy", "calm", "anxious", "sad"])  # Simulated emotion detection
    return pitch, emotion


def generate_meditation_script(emotion):
    prompt = f"Create a guided meditation script for someone who feels {emotion}."
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


def play_background_sound(emotion):
    if emotion == "calm":
        sound_file = background_sounds["calm"]
    elif emotion == "anxious":
        sound_file = background_sounds["nature"]
    else:
        sound_file = background_sounds["ocean"]

    # Play the sound file (pseudo-code, replace with actual sound playing method)
    print(f"Playing background sound: {sound_file}")


def provide_guided_meditation(script):
    engine.say(script)
    engine.runAndWait()


def listen_and_meditate():
    with sr.Microphone() as source:
        print("Please speak your feelings...")
        audio_data = recognizer.listen(source)
        print("Analyzing speech...")

        # Analyze speech for pitch and emotion
        pitch, emotion = analyze_speech(audio_data)

        # Generate meditation script based on user emotion
        meditation_script = generate_meditation_script(emotion)

        # Play background sound based on user emotion
        play_background_sound(emotion)

        # Provide guided meditation audio
        provide_guided_meditation(meditation_script)


if __name__ == "__main__":
    print("Immersive Audio Experience for Guided Meditation")
    while True:
        listen_and_meditate()
        time.sleep(1)  # Delay before listening again
