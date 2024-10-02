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

# Function to generate contextual language learning responses
def generate_response(user_input):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"You are a language learning assistant. Respond to the user's input: {user_input}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Read the phrase aloud
def speak_phrase(phrase):
    engine.say(phrase)
    engine.runAndWait()

# Real-time voice detection for language learning
def listen_and_learn():
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        print(f"You said: {user_input}")

        # Generate and respond with a contextual language learning dialogue
        dialogue_response = generate_response(user_input)
        print(f"Assistant: {dialogue_response}")

        # Read the response aloud
        speak_phrase(dialogue_response)

# Main function to run the audio-enhanced language learning assistant
def audio_enhanced_language_learning_assistant():
    print("Welcome to the Audio-Enhanced Conversational AI for Language Practice!")
    print("Feel free to ask about phrases, vocabulary, or any language-related topics.")
    while True:
        listen_and_learn()

# Run the language learning assistant
if __name__ == "__main__":
    audio_enhanced_language_learning_assistant()
