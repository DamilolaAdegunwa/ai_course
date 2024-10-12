import os
import time
import numpy as np
import librosa
import soundfile as sf
from openai import OpenAI
from apikey import apikey  # Assuming your API key is stored in apikey.py
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
client = OpenAI()

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Background sounds
background_sounds = {
    "happy": "happy_music.wav",
    "sad": "sad_music.wav",
    "neutral": "neutral_music.wav",
}

# Load emotion detection model
with open('emotion_model.pkl', 'rb') as file:
    emotion_model = pickle.load(file)


def detect_emotion_from_audio(audio_data):
    # Extract features from the audio
    audio_array = np.array(audio_data.get_array_of_samples())
    mfccs = librosa.feature.mfcc(y=audio_array.astype(float), sr=44100, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

    # Normalize the features
    scaler = StandardScaler()
    mfccs_scaled = scaler.fit_transform(mfccs_mean)

    # Predict emotion
    emotion = emotion_model.predict(mfccs_scaled)
    return emotion[0]


def play_background_sound(emotion):
    if emotion in background_sounds:
        sound_file = background_sounds[emotion]
        sound = AudioSegment.from_file(sound_file)
        play(sound)


def listen_and_recognize():
    with sr.Microphone() as source:
        print("Listening for user input...")
        audio_data = recognizer.listen(source)
        return audio_data


def interactive_emotional_responses():
    print("Welcome to the Real-Time Emotion Recognition Experience!")

    while True:
        audio_data = listen_and_recognize()
        detected_emotion = detect_emotion_from_audio(audio_data)
        print(f"Detected emotion: {detected_emotion}")

        # Generate response based on detected emotion
        if detected_emotion == "happy":
            engine.say("I'm glad to hear you're feeling happy! Let's celebrate with some music!")
            engine.runAndWait()
            play_background_sound("happy")
        elif detected_emotion == "sad":
            engine.say("I'm here for you. Let's take a moment to relax.")
            engine.runAndWait()
            play_background_sound("sad")
        else:
            engine.say("It's okay to feel neutral. Let's enjoy some ambient sounds.")
            engine.runAndWait()
            play_background_sound("neutral")


if __name__ == "__main__":
    interactive_emotional_responses()
