import os
from openai import OpenAI
from apikey import apikey  # Your apikey.py file with the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

import pyttsx3
import speech_recognition as sr

# TTS Engine Initialization
engine = pyttsx3.init()

# Speech Recognition Engine Initialization
recognizer = sr.Recognizer()

# Translate user input to desired language
def translate_text(text, target_language):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Translate the following text to {target_language}: {text}",
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Generate audio summary of the translated text
def generate_audio_summary(text, language):
    summary_prompt = f"Provide a concise summary of the following text in {language}: {text}"
    summary_response = client.completions.create(
        model="text-davinci-003",
        prompt=summary_prompt,
        max_tokens=100
    )
    summary_text = summary_response.choices[0].text.strip()
    engine.say(summary_text)
    engine.runAndWait()

# Real-time voice detection and translation
def listen_and_translate(target_language):
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        print(f"You said: {user_input}")

        # Translate the user input
        translated_text = translate_text(user_input, target_language)
        print(f"Translated text: {translated_text}")

        # Generate and play audio summary
        generate_audio_summary(translated_text, target_language)

# Main function to start the translation and summarization experience
def interactive_translation():
    target_language = input("Enter the target language (e.g., Spanish, French): ")
    while True:
        listen_and_translate(target_language)

# Run the interactive translation experience
if __name__ == "__main__":
    interactive_translation()
