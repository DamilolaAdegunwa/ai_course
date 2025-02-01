import os
from openai import OpenAI
from apikey import apikey  # Ensure your API key is stored in apikey.py
import pyttsx3
import speech_recognition as sr
import random
from playsound import playsound

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# TTS Engine Initialization
engine = pyttsx3.init()

# Speech Recognition Engine Initialization
recognizer = sr.Recognizer()

# Game state variables
game_state = {
    "location": "start",
    "inventory": []
}


# Load audio files for sound effects
def load_audio_effect(effect_name):
    return f"audio/{effect_name}.wav"  # Assuming audio files are in an "audio" folder


# Generate narrative based on user input
def generate_narrative(user_input):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"The player is currently at {game_state['location']}. User input: '{user_input}'. What happens next?",
        max_tokens=150
    )
    return response.choices[0].text.strip()


# Real-time voice detection and response
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for your action...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        return user_input.lower()


# Play audio effects
def play_audio(effect_name):
    audio_file = load_audio_effect(effect_name)
    playsound(audio_file)


# Main game loop
def interactive_audio_game():
    while True:
        # Prompt the player
        if game_state["location"] == "start":
            prompt = "You find yourself at the entrance of a mysterious cave. What do you want to do?"
        else:
            prompt = f"You are at {game_state['location']}. What do you want to do?"

        engine.say(prompt)
        engine.runAndWait()

        # Listen for user input
        user_input = listen_and_recognize()
        print(f"You said: {user_input}")

        # Generate narrative based on user input
        narrative = generate_narrative(user_input)
        print(f"Game says: {narrative}")

        # Play sound effects based on user actions
        if "explore" in user_input:
            game_state["location"] = "cave"
            play_audio("dripping_water")
        elif "light my torch" in user_input:
            game_state["inventory"].append("torch")
            play_audio("torch_lit")
        elif "open treasure" in user_input and "treasure" in game_state["location"]:
            play_audio("treasure_opened")
            game_state["inventory"].append("gold")
            narrative += " You found gold inside the treasure chest!"

        engine.say(narrative)
        engine.runAndWait()


# Run the interactive audio game
if __name__ == "__main__":
    interactive_audio_game()
