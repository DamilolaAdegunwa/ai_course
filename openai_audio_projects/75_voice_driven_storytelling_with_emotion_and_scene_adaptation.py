import os
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

# Background sounds for different emotions
background_sounds = {
    "happy": "heroic_music.wav",
    "sad": "melancholic_music.wav",
    "fear": "suspenseful_music.wav",
    "neutral": "ambient_music.wav",
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
        print("Listening for your input...")
        audio_data = recognizer.listen(source)
        return audio_data


def generate_story_part(user_input, detected_emotion):
    prompt = f"The user is feeling {detected_emotion}. They said: '{user_input}'. What happens next in the story?"
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()


def interactive_storytelling():
    print("Welcome to the Voice-Driven Storytelling Experience!")

    while True:
        audio_data = listen_and_recognize()
        detected_emotion = detect_emotion_from_audio(audio_data)
        print(f"Detected emotion: {detected_emotion}")

        user_input = recognizer.recognize_google(audio_data)
        print(f"You said: {user_input}")

        # Generate response based on detected emotion
        story_part = generate_story_part(user_input, detected_emotion)
        engine.say(story_part)
        engine.runAndWait()

        # Play background sound based on detected emotion
        play_background_sound(detected_emotion)


if __name__ == "__main__":
    interactive_storytelling()
