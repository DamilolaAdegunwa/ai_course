import os
import openai
import sounddevice as sd
import numpy as np
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
import simpleaudio as sa
from pydub import AudioSegment

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Story data structure
story_data = {
    "start": {
        "text": "Welcome to the enchanted forest. You see two paths: one leading to a dark cave and the other to a bright meadow. Which path do you choose?",
        "options": {
            "cave": "You step into the dark cave and hear a low growl.",
            "meadow": "You walk into the bright meadow and see beautiful flowers."
        }
    },
    "cave": {
        "text": "A fierce dragon appears! Do you want to fight or flee?",
        "options": {
            "fight": "You bravely confront the dragon!",
            "flee": "You run back to the safety of the forest."
        }
    },
    "meadow": {
        "text": "A friendly fairy offers you a magical gift. Do you accept or refuse?",
        "options": {
            "accept": "You accept the gift and feel empowered.",
            "refuse": "You politely refuse and continue exploring."
        }
    },
}


# Function to generate audio for story narration
def generate_story_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "story_audio.wav"
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


# Main function to run the storytelling platform
def run_storytelling():
    print("Welcome to the Interactive Voice-Activated Storytelling Platform!")

    current_scene = "start"

    while True:
        # Narrate the current scene
        scene = story_data[current_scene]
        generate_story_audio(scene["text"])

        # Present options to the user
        options_text = "\n".join([f"{key}: {value}" for key, value in scene["options"].items()])
        print("Options:", options_text)
        generate_story_audio(options_text)

        # Listen for user input
        user_input = recognize_speech()
        print("You said:", user_input)

        # Update the current scene based on user choice
        if user_input in scene["options"]:
            current_scene = user_input
        else:
            print("Sorry, I didn't catch that. Please choose one of the options.")
            generate_story_audio("Sorry, I didn't catch that. Please choose one of the options.")

        # Exit condition
        if current_scene == "exit":
            print("Thank you for playing!")
            break


# Main entry point
if __name__ == "__main__":
    run_storytelling()
