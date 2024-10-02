import os
from openai import OpenAI
from apikey import apikey  # API key stored in a separate file
import pyttsx3
import speech_recognition as sr
import random
from pydub import AudioSegment
import simpleaudio as sa

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# TTS Engine Initialization
engine = pyttsx3.init()

# Speech Recognition Engine Initialization
recognizer = sr.Recognizer()

# Function to generate audio for narration
def generate_audio(text, filename="adventure_audio.wav"):
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# Function to load and play audio
def play_audio(file_path):
    sound = AudioSegment.from_file(file_path)
    play_obj = sa.play_buffer(sound.raw_data, num_channels=sound.channels,
                              bytes_per_sample=sound.sample_width,
                              sample_rate=sound.frame_rate)
    play_obj.wait_done()

# Generate AI-driven narrative and respond to user input
def generate_adventure_response(user_input):
    prompt = f"The user says: '{user_input}'. What happens next in the adventure?"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response['choices'][0]['message']['content'].strip()

# Real-time voice detection and response
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for your command...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            return user_input.lower()
        except sr.UnknownValueError:
            print("Sorry, I could not understand that.")
            return ""
        except sr.RequestError:
            print("Could not request results; check your network connection.")
            return ""

# Main function to run the guided exploration
def run_audio_adventure():
    print("Welcome to the AI Audio Adventure!")
    intro_text = "You find yourself at the entrance of a mysterious forest. What do you want to do?"
    generate_audio(intro_text)
    play_audio(generate_audio(intro_text))

    while True:
        user_input = listen_and_recognize()
        if user_input:
            adventure_response = generate_adventure_response(user_input)
            print("Adventure Response:", adventure_response)
            adventure_audio_file = generate_audio(adventure_response)
            play_audio(adventure_audio_file)

            # Check if the user wants to continue
            if "exit" in user_input:
                farewell_text = "Thank you for playing! Until next time!"
                generate_audio(farewell_text)
                play_audio(generate_audio(farewell_text))
                break

# Run the audio adventure experience
if __name__ == "__main__":
    run_audio_adventure()
