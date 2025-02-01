import os
from openai import OpenAI
from apikey import apikey  # Ensure your API key is stored in apikey.py
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
import numpy as np
import soundfile as sf

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# TTS Engine Initialization
engine = pyttsx3.init()

# Speech Recognition Engine Initialization
recognizer = sr.Recognizer()

# Background sound settings
background_sounds = {
    "relaxation": "relaxing_sound.wav",
    "mindfulness": "mindfulness_sound.wav",
    "visualization": "visualization_sound.wav"
}


# Load audio files for background sound
def load_background_sound(file_path):
    return AudioSegment.from_file(file_path)


# Play background sound
def play_background_sound(sound):
    sound_segment = load_background_sound(sound)
    sound_segment.export("temp_sound.wav", format="wav")
    os.system("start temp_sound.wav")  # Play the sound using the default media player


# Generate a meditation script based on user mood
def generate_meditation_script(style):
    prompt = f"Generate a {style} meditation script."
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()


# Real-time voice detection and response
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for your command...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        print(f"You said: {user_input}")
        return user_input


# Main meditation assistant loop
def meditation_assistant():
    print("Welcome to your Interactive Meditation Assistant!")

    while True:
        # Listen for user commands
        user_input = listen_and_recognize()

        if "relaxing" in user_input:
            # Start relaxing meditation session
            print("Starting relaxation meditation.")
            play_background_sound(background_sounds["relaxation"])
            script = generate_meditation_script("relaxation")
            engine.say(script)
            engine.runAndWait()

        elif "mindfulness" in user_input:
            # Start mindfulness meditation session
            print("Starting mindfulness meditation.")
            play_background_sound(background_sounds["mindfulness"])
            script = generate_meditation_script("mindfulness")
            engine.say(script)
            engine.runAndWait()

        elif "visualization" in user_input:
            # Start visualization meditation session
            print("Starting visualization meditation.")
            play_background_sound(background_sounds["visualization"])
            script = generate_meditation_script("visualization")
            engine.say(script)
            engine.runAndWait()

        elif "stop" in user_input:
            # Stop the current session
            print("Stopping the meditation session.")
            os.system("taskkill /im wmplayer.exe /f")  # Close the media player
            engine.say("Meditation session ended. Thank you!")
            engine.runAndWait()
            break

        else:
            print("I didn't understand that. Please say 'relaxing', 'mindfulness', 'visualization', or 'stop'.")


# Run the meditation assistant
if __name__ == "__main__":
    meditation_assistant()
