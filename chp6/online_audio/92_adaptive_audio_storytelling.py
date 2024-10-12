import os
from openai import OpenAI
from apikey import apikey  # Store your OpenAI API key here

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

import pyttsx3
import speech_recognition as sr
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile

# TTS Engine Initialization
engine = pyttsx3.init()

# Speech Recognition Engine Initialization
recognizer = sr.Recognizer()

# Function to analyze voice tone
def analyze_voice_tone(audio_data):
    # Convert audio data to a waveform
    sample_rate, samples = wavfile.read(audio_data)
    # Analyze the audio data for tone (simplified for this example)
    tone = "neutral"  # Placeholder for actual emotion recognition logic
    # Implement a real emotion detection method based on audio analysis (not shown)
    return tone

# Generate AI-driven narrative and character interaction
def generate_story(input_prompt):
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=input_prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Generate music based on emotional tone
def generate_music(tone):
    if tone == "happy":
        return "upbeat_music.wav"
    elif tone == "fearful":
        return "eerie_music.wav"
    else:
        return "neutral_music.wav"

# Load audio files for spatial sound
def load_audio(file_path):
    return AudioSegment.from_file(file_path)

# Real-time voice detection and response
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
        return audio

# Play background music based on the user's emotional state
def play_background_music(tone):
    music_file = generate_music(tone)
    music = load_audio(music_file)
    play(music)

# Main story progression and interaction loop
def interactive_storytelling():
    while True:
        # Narrate the current story
        story_prompt = "You find yourself in a mysterious cave. What do you do next?"
        story_part = generate_story(story_prompt)
        engine.say(story_part)
        engine.runAndWait()

        # Listen for user input
        audio_data = listen_and_recognize()
        tone = analyze_voice_tone(audio_data)  # Analyze the tone of voice

        # Play background music based on detected tone
        play_background_music(tone)

        # Respond to user input to influence the story
        user_input = recognizer.recognize_google(audio_data)
        if "continue" in user_input:
            continue
        elif "scared" in user_input:
            engine.say("You feel a chill in the air...")
        elif "exciting" in user_input:
            engine.say("You find a treasure chest!")

        engine.runAndWait()

# Run the interactive storytelling experience
if __name__ == "__main__":
    interactive_storytelling()
