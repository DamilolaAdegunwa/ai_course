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

# Function to generate language learning dialogues
def generate_learning_dialogue(user_input):
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

        # Generate and respond with language learning dialogue
        dialogue_response = generate_learning_dialogue(user_input)
        print(f"Assistant: {dialogue_response}")

        # Read the response aloud
        speak_phrase(dialogue_response)

# Main function to run the language learning assistant
def dynamic_language_learning_assistant():
    print("Welcome to the Dynamic Language Learning Assistant!")
    print("You can ask about greetings, verb conjugations, or any language-related questions.")
    while True:
        listen_and_learn()

# Run the language learning assistant
if __name__ == "__main__":
    dynamic_language_learning_assistant()
