import os
import random
import time
from openai import OpenAI
from apikey import apikey  # Assuming your API key is stored in apikey.py
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
client = OpenAI()

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Background soundscapes
background_sounds = {
    "cave": "cave_ambience.wav",
    "forest": "forest_ambience.wav",
    "spooky": "spooky_soundtrack.wav",
}


def generate_story_prompt(user_choice):
    prompt = f"Create a story segment based on the choice: {user_choice}. Include dialogue and descriptive elements."
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


def play_background_sound(context):
    if context in background_sounds:
        sound_file = background_sounds[context]
        sound = AudioSegment.from_file(sound_file)
        play(sound)


def listen_and_interact():
    with sr.Microphone() as source:
        print("Please state your choice...")
        audio_data = recognizer.listen(source)
        print("Analyzing choice...")

        try:
            user_choice = recognizer.recognize_google(audio_data)
            print(f"You said: {user_choice}")
            return user_choice
        except sr.UnknownValueError:
            print("Sorry, I could not understand you.")
            return None


def interactive_storytelling():
    print("Welcome to the Interactive Storytelling Experience!")

    # Start with an initial story setting
    initial_choice = "You find yourself at a crossroads in a dark forest. Do you want to go left towards the spooky cave or right towards the bright meadow?"
    print(initial_choice)

    # Play initial ambient sound
    play_background_sound("forest")

    while True:
        user_choice = listen_and_interact()
        if user_choice:
            story_segment = generate_story_prompt(user_choice)
            engine.say(story_segment)
            engine.runAndWait()

            # Play sound based on user choice
            if "cave" in user_choice:
                play_background_sound("cave")
            elif "meadow" in user_choice:
                play_background_sound("forest")
            else:
                play_background_sound("spooky")


if __name__ == "__main__":
    interactive_storytelling()
