import os
import openai
import sounddevice as sd
import numpy as np
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
import simpleaudio as sa
from pydub import AudioSegment
import random

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Meditation data structure
meditation_data = {
    "relaxation": [
        "Let's begin with a deep breath. Inhale slowly... and exhale.",
        "Imagine a peaceful place. What do you see, hear, and feel?",
        "Allow any tension in your body to melt away.",
    ],
    "focus": [
        "Close your eyes and take a moment to center your thoughts.",
        "Focus on your breathing, letting go of distractions.",
        "Visualize your goals and what you want to achieve.",
    ],
    "sleep": [
        "Get comfortable and take a few deep breaths.",
        "Imagine a gentle wave lapping at the shore, soothing and calming.",
        "As you listen to my voice, allow your body to relax into sleep.",
    ],
}

# Background sounds
background_sounds = {
    "ocean": "ocean.wav",
    "forest": "forest.wav",
    "rain": "rain.wav",
}


# Function to generate audio for meditation guidance
def generate_meditation_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "meditation_audio.wav"
    tts.save(audio_file)

    # Play the audio using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(audio_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until audio is finished playing


# Function to record user speech
def record_audio(duration=5):
    print("Recording...")
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return recording


# Function to recognize speech and return the text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service."


# Function to play background sound
def play_background_sound(sound_name):
    if sound_name in background_sounds:
        sound_file = background_sounds[sound_name]
        sound = AudioSegment.from_wav(sound_file)
        play_obj = sa.play_buffer(sound.raw_data, num_channels=sound.channels,
                                  samplerate=sound.frame_rate,
                                  blocksize=sound.frame_rate)
        return play_obj


# Main function to run the meditation assistant
def run_meditation_assistant():
    print("Welcome to the AI-Powered Meditation Assistant!")

    while True:
        # Ask the user for their preference
        generate_meditation_audio("What type of meditation would you like? Relaxation, focus, or sleep?")

        user_input = recognize_speech()
        print("You said:", user_input)

        if user_input in meditation_data:
            generate_meditation_audio("Great choice! Let's begin your meditation.")
            # Play background sound
            play_background_sound("ocean")  # Example, this can be user-defined
            for line in meditation_data[user_input]:
                generate_meditation_audio(line)
        else:
            print("Sorry, I didn't catch that. Please choose relaxation, focus, or sleep.")
            generate_meditation_audio("Sorry, I didn't catch that. Please choose relaxation, focus, or sleep.")

        # Exit condition
        if user_input == "exit":
            print("Thank you for using the meditation assistant! Goodbye.")
            break


# Main entry point
if __name__ == "__main__":
    run_meditation_assistant()
