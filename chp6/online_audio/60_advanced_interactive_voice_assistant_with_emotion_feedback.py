"""
Project Title: Advanced Interactive Voice Assistant with Emotion-Based Audio Feedback
File Name: advanced_interactive_voice_assistant_with_emotion_feedback.py

Project Description:
This project takes your OpenAI audio integration skills to the next level by creating an interactive voice assistant that responds to user input with real-time emotion-based audio feedback. The assistant can detect user emotion through input, synthesize voice responses, and adjust the emotional tone of background music dynamically based on the conversation. Additionally, it introduces advanced speech recognition, sentiment analysis, and a more complex 3D audio environment.

Example Use Cases:
Input: "I'm feeling great today!"
Expected Output: Upbeat response with cheerful background music.
Input: "I'm really stressed out."
Expected Output: Calming response with soft, soothing background music.
Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # OpenAI API key stored in this file
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
import numpy as np
import random
import librosa
from transformers import pipeline  # For sentiment analysis
from pydub.playback import play  # Add this import
# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS Engine
engine = pyttsx3.init()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Sentiment Analysis model initialization (using Huggingface)
sentiment_analyzer = pipeline("sentiment-analysis")

# Sound source positions for 3D audio
sound_sources = {
    "music": {"x": 1, "y": 0, "z": 0},
    "environment": {"x": 0, "y": 1, "z": 0}
}


# Generate background music based on sentiment
def generate_music(emotion):
    if emotion == "POSITIVE":
        return "happy_music.wav"
    elif emotion == "NEGATIVE":
        return "calming_music.wav"
    else:
        return "neutral_music.wav"


# Load audio and apply spatial audio effect
def load_audio(file_path):
    return AudioSegment.from_file(file_path)


def apply_spatial_audio(sound, x, y, z):
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    volume_adjusted = sound - (distance * 5)
    return volume_adjusted


# Detect and recognize user speech
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        user_input = recognizer.recognize_google(audio)
        return user_input


# Analyze sentiment of the user input
def analyze_sentiment(user_input):
    sentiment_result = sentiment_analyzer(user_input)
    return sentiment_result[0]['label']


# Generate response from OpenAI
def generate_response(input_text):
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=input_text,
        max_tokens=150
    )
    return response.choices[0].text.strip()


# Play real-time music based on emotion
def play_music_based_on_emotion(emotion):
    music_file = generate_music(emotion)
    music = load_audio(music_file)
    spatial_music = apply_spatial_audio(music, sound_sources["music"]["x"], sound_sources["music"]["y"],
                                        sound_sources["music"]["z"])
    play(spatial_music)


# Interactive Voice Assistant loop
def voice_assistant():
    while True:
        user_input = listen_and_recognize()
        print(f"User said: {user_input}")

        # Analyze sentiment
        sentiment = analyze_sentiment(user_input)
        print(f"Detected sentiment: {sentiment}")

        # Generate response
        response = generate_response(f"The user said: {user_input}. Respond accordingly.")
        print(f"AI response: {response}")

        # Speak the AI response
        engine.say(response)
        engine.runAndWait()

        # Play background music based on sentiment
        play_music_based_on_emotion(sentiment)


# Main function
if __name__ == "__main__":
    voice_assistant()
"""
Example Inputs and Expected Outputs:
Example Input 1:

User Input: "I just had the best day!"
Sentiment Detected: Positive
Expected AI Response: "That's fantastic! I'm so happy to hear that."
Background Music: Upbeat and joyful music.
Example Input 2:

User Input: "I feel so overwhelmed and stressed."
Sentiment Detected: Negative
Expected AI Response: "I'm sorry you're feeling that way. Take a deep breath, it'll be okay."
Background Music: Soft and calming music.
Features Added (Compared to Previous Project):
Sentiment Analysis: Detects user emotion and adjusts background audio accordingly.
Real-time Emotional Feedback: Background music adapts to the user's emotional state in real-time.
Advanced Response Generation: AI dynamically generates meaningful responses based on user input.
Spatial Audio: Uses 3D sound effects to enhance the immersive experience.
This project pushes the boundaries of your previous audio integration project by introducing emotional intelligence and sentiment-driven interaction while maintaining ease of understanding and testing.
"""
