import os
from openai import OpenAI
from apikey import apikey  # Ensure your API key is stored in apikey.py
import pyttsx3
import speech_recognition as sr

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# TTS Engine Initialization
engine = pyttsx3.init()

# Speech Recognition Engine Initialization
recognizer = sr.Recognizer()


# Detect the language of the user's speech
def detect_language(user_input):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"What language is this: '{user_input}'?",
        max_tokens=10
    )
    return response.choices[0].text.strip()


# Provide language learning content based on detected language
def provide_learning_content(language):
    if language.lower() == "french":
        return "Let's practice some French phrases. Repeat after me: 'Comment ça va?'"
    elif language.lower() == "spanish":
        return "Vamos a practicar algunas frases en español. Repite después de mí: '¿Cómo estás?'"
    else:
        return "Let's practice some English phrases. Repeat after me: 'Hello, how are you?'"


# Real-time voice detection and response
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        return user_input


# Interactive language learning loop
def interactive_language_learning():
    while True:
        user_input = listen_and_recognize()
        print(f"You said: {user_input}")

        language = detect_language(user_input)
        print(f"Detected language: {language}")

        learning_content = provide_learning_content(language)
        print(f"Assistant says: {learning_content}")

        # Text-to-Speech
        engine.say(learning_content)
        engine.runAndWait()


# Run the interactive language learning assistant
if __name__ == "__main__":
    interactive_language_learning()
