import os
import openai
import pyttsx3
import random
import time
import sounddevice as sd
import numpy as np

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define meditation themes and background sounds
themes = {
    "relaxation": {
        "script": "Welcome to your relaxation meditation. Take a deep breath in... and out. Feel the tension leaving your body.",
        "sounds": "relaxation_sound.wav"
    },
    "focus": {
        "script": "Welcome to your focus meditation. Sit comfortably and concentrate on your breath. Inhale... Exhale...",
        "sounds": "focus_sound.wav"
    },
    "stress relief": {
        "script": "Welcome to your stress relief meditation. Picture a serene landscape... feel the stress melting away.",
        "sounds": "stress_relief_sound.wav"
    }
}

# Function to generate meditation script based on mood
def generate_meditation_script(theme):
    prompt = f"Create a meditation script focused on {theme}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']

# Function to play background sounds
def play_background_sound(sound_file, duration):
    # Load sound file as a numpy array
    # For this example, we will create a simple sine wave as placeholder audio
    frequency = 440  # Frequency in Hertz (A4)
    fs = 44100  # Sampling frequency
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Play the generated audio
    sd.play(audio, fs)
    sd.wait()  # Wait until sound has finished playing

# Function to speak out text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to get user preferences
def get_user_preferences():
    mood = input("How do you feel? (relaxed, focused, stressed): ").lower()
    duration = int(input("How long do you want to meditate (in minutes)? "))
    return mood, duration

# Function to play meditation session
def play_meditation_session():
    mood, duration = get_user_preferences()
    if mood in themes:
        theme = mood
        meditation_script = generate_meditation_script(theme)
        background_sound = themes[theme]["sounds"]

        # Speak the meditation script
        speak(meditation_script)

        # Play background sound for the specified duration
        play_background_sound(background_sound, duration * 60)  # Convert minutes to seconds
    else:
        print("Sorry, I didn't understand that mood. Please choose relaxed, focused, or stressed.")

# Main function to start the meditation session
if __name__ == "__main__":
    print("Welcome to the Adaptive Audio Meditation Guide!")
    play_meditation_session()
