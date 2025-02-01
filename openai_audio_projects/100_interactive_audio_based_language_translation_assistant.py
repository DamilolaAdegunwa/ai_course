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


# Function to translate user input
def translate_text(user_input, target_language):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Translate the following text into {target_language}: {user_input}",
        max_tokens=60
    )
    return response.choices[0].text.strip()


# Read the translated text aloud
def speak_translation(translation):
    engine.say(translation)
    engine.runAndWait()


# Real-time voice detection for translation
def listen_and_translate():
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        print(f"You said: {user_input}")

        # Determine the target language based on user input
        if "in Spanish" in user_input:
            target_language = "Spanish"
        elif "in French" in user_input:
            target_language = "French"
        else:
            print("Language not recognized. Please specify either 'Spanish' or 'French'.")
            return

        # Remove the target language instruction for translation
        text_to_translate = user_input.split("in")[0].strip()

        # Generate the translation
        translation = translate_text(text_to_translate, target_language)
        print(f"Translation in {target_language}: {translation}")

        # Read the translation aloud
        speak_translation(translation)


# Main function to run the interactive audio translation assistant
def interactive_audio_translation_assistant():
    print("Welcome to the Interactive Audio-Based Language Translation Assistant!")
    print("You can ask me to translate phrases into Spanish or French.")
    while True:
        listen_and_translate()


# Run the language translation assistant
if __name__ == "__main__":
    interactive_audio_translation_assistant()
