"""
Project Title: Advanced Interactive Audio Narration with Dynamic Emotion Detection
File Name: advanced_interactive_audio_with_emotion_detection.py

Project Description:
This project extends your existing audio interaction capabilities by incorporating dynamic emotion detection from user speech input. Based on the user's speech tone (emotion), the OpenAI API generates an emotionally responsive story, while background music adapts in real-time to match the detected emotion. The system provides a more immersive, emotionally attuned storytelling experience, including interactive narrative progression and real-time audio transformation.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # The apikey is stored in apikey.py
import speech_recognition as sr
import pyttsx3
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS and speech recognition
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Audio positions for 3D sound
sound_sources = {
    "narration": {"x": 0, "y": 0, "z": 0},
    "background": {"x": 1, "y": 0, "z": 0}
}

# Emotion-based music generator
def generate_music(emotion):
    if emotion == "happy":
        return "happy_tune.wav"
    elif emotion == "sad":
        return "sad_tune.wav"
    else:
        return "neutral_tune.wav"

# Load audio
def load_audio(file_path):
    return AudioSegment.from_file(file_path)

# Spatial audio adjustment
def apply_spatial_audio(sound, x, y, z):
    distance = np.sqrt(x**2 + y**2 + z**2)
    volume_adjusted = sound - (distance * 5)
    return volume_adjusted

# Emotion detection (mockup, real implementation can use pre-trained models)
def detect_emotion(user_speech):
    # For simplicity, we'll randomly pick an emotion here
    return np.random.choice(["happy", "sad", "neutral"])

# Generate AI-driven story based on emotion
def generate_emotional_story(emotion):
    prompt = f"Tell a story that reflects a {emotion} mood."
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Speech recognition to capture user input
def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            return user_input
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."

# Main interactive loop with emotion-based response
def interactive_audio_story():
    while True:
        # Listen to the user's input and detect their emotional tone
        user_input = listen_and_recognize()
        emotion = detect_emotion(user_input)

        # Generate emotionally responsive story
        story = generate_emotional_story(emotion)
        print(f"Story: {story}")
        engine.say(story)
        engine.runAndWait()

        # Generate music based on the detected emotion
        music_file = generate_music(emotion)
        background_music = load_audio(music_file)
        spatial_music = apply_spatial_audio(background_music, sound_sources["background"]["x"], sound_sources["background"]["y"], sound_sources["background"]["z"])
        play(spatial_music)

        if "end" in user_input.lower():
            print("Ending the session.")
            break

# Start the interactive storytelling
if __name__ == "__main__":
    interactive_audio_story()
"""
Example Inputs and Outputs:
Input 1 (User Speech):
"I feel great, tell me a story."
Detected Emotion: Happy
Generated Story:
"Once upon a time, in a land full of joy and laughter, there lived a young prince who couldn't stop smiling...
Background Music: Upbeat, cheerful music plays in the background.

Input 2 (User Speech):
"I'm feeling down today."
Detected Emotion: Sad
Generated Story:
"On a rainy day in a quiet town, a lonely traveler wandered the streets, reflecting on lost moments and forgotten dreams..."
Background Music: Somber, melancholic music plays.

Key Features:
Emotion Detection: Detects emotions from user input (mockup, easily extendable with models like OpenAI's Whisper for advanced sentiment analysis).
Real-time Emotionally Responsive Storytelling: AI-generated stories dynamically change based on the detected emotion.
Dynamic Background Music: Emotion-based music changes the atmosphere of the story in real time.
3D Audio Simulation: Applies spatial audio effects for an immersive sound environment.
This project offers a richer, more immersive interactive audio experience than your previous project by incorporating emotion detection and corresponding narrative generation with dynamic sound adjustments.
"""