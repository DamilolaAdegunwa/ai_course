import os
import openai
import sounddevice as sd
import numpy as np
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
import playsound
import json

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to generate audio for lessons
def generate_lesson_audio(text, language='es'):
    tts = gTTS(text=text, lang=language)
    audio_file = "lesson_audio.mp3"
    tts.save(audio_file)
    playsound.playsound(audio_file)


# Function to record user speech
def record_audio(duration=5):
    print("Recording...")
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return recording


# Function to recognize speech and provide feedback
def recognize_speech(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service."


# Function to simulate conversation
def engage_in_conversation(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']


# Main function to run the language learning platform
def run_language_learning():
    print("Welcome to the Interactive Audio-Based Language Learning Platform!")
    language = input("Choose a language (e.g., 'es' for Spanish): ")

    while True:
        lesson = input("What lesson would you like to learn? (type 'exit' to quit): ")
        if lesson.lower() == 'exit':
            break

        # Generate lesson audio
        lesson_text = f"Welcome to the {lesson} lesson. Here are some greetings: Hola, Buenos días, Buenas tardes, Buenas noches."
        generate_lesson_audio(lesson_text, language)

        # Record user pronunciation
        audio_data = record_audio()
        audio_file = "user_pronunciation.wav"
        sf.write(audio_file, audio_data, 44100)

        # Recognize user speech
        recognized_text = recognize_speech(audio_file)
        print("You said:", recognized_text)

        # Simulate a conversation
        if recognized_text.lower() in ["hola", "buenos días", "buenas tardes", "buenas noches"]:
            conversation_input = "How would you respond to a greeting in Spanish?"
            response = engage_in_conversation(conversation_input)
            print("AI:", response)


# Main entry point
if __name__ == "__main__":
    run_language_learning()
