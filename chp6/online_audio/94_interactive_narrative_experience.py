import os
from openai import OpenAI
from apikey import apikey  # Store your OpenAI API key in this file

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
import numpy as np
import random

# TTS Engine Initialization
engine = pyttsx3.init()

# Speech Recognition Engine Initialization
recognizer = sr.Recognizer()

# Load audio files for emotional music
def load_audio(file_path):
    return AudioSegment.from_file(file_path)

# Generate AI-driven narrative based on user input
def generate_story(input_prompt):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=input_prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Real-time voice detection and response
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        return user_input

# Play music based on user-defined emotional context
def play_emotional_music(emotion):
    # Determine music file based on emotion
    music_files = {
        "happy": "happy_music.wav",
        "sad": "sad_music.wav",
        "suspense": "suspense_music.wav",
        "victory": "victory_music.wav"
    }
    music_file = music_files.get(emotion, "default_music.wav")
    music = load_audio(music_file)
    # Adjust and play the music
    music.export("current_music.wav", format="wav")
    os.system("start current_music.wav")  # Plays the music

# Main narrative loop
def interactive_storytelling():
    print("Welcome to the Interactive Narrative Experience!")
    initial_prompt = "Once upon a time, in a land filled with magic and wonder, the adventurer stood at the edge of a dark forest. What happens next?"
    story_part = generate_story(initial_prompt)
    engine.say(story_part)
    engine.runAndWait()

    while True:
        user_input = listen_and_recognize()
        print(f"You said: {user_input}")

        if "talk to the wizard" in user_input:
            response = generate_story("What does the wizard say?")
            engine.say(response)
            engine.runAndWait()
        elif "happy ending" in user_input:
            response = generate_story("The adventurer wishes for a happy ending.")
            engine.say(response)
            engine.runAndWait()
            play_emotional_music("happy")
        elif "sad ending" in user_input:
            response = generate_story("The adventurer meets a tragic fate.")
            engine.say(response)
            engine.runAndWait()
            play_emotional_music("sad")
        elif "change the music" in user_input:
            play_emotional_music(random.choice(["happy", "sad", "suspense", "victory"]))
        else:
            # Continue the story with the user input
            additional_prompt = f"Then {user_input} happened."
            story_part = generate_story(additional_prompt)
            engine.say(story_part)
            engine.runAndWait()

# Run the interactive storytelling experience
if __name__ == "__main__":
    interactive_storytelling()
