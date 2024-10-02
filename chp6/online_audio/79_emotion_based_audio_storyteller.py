import os
import openai
import sounddevice as sd
import numpy as np
import pyttsx3
from scipy.io.wavfile import write
import joblib  # For loading the emotion recognition model
import pydub

# Load pre-trained emotion recognition model
emotion_model = joblib.load('path/to/emotion_recognition_model.pkl')

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize text-to-speech engine
engine = pyttsx3.init()


# Function to record audio from the user
def record_audio(duration=5):
    print("Recording your voice... Speak now!")
    fs = 44100  # Sample rate
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('user_input.wav', fs, myrecording)  # Save as WAV file
    return 'user_input.wav'


# Function to detect emotion from recorded audio
def detect_emotion(audio_file):
    # This is a placeholder function
    # Implement your own feature extraction and prediction logic here
    features = extract_features(audio_file)  # Replace with actual feature extraction
    emotion = emotion_model.predict([features])  # Predict emotion
    return emotion[0]


# Function to generate a story based on the detected emotion
def generate_story(emotion):
    prompt = f"Create a story suitable for someone feeling {emotion}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response['choices'][0]['message']['content']


# Function to narrate the story with text-to-speech
def narrate_story(story):
    engine.say(story)
    engine.runAndWait()


# Function to add background music based on emotion
def add_background_music(emotion):
    if emotion == 'happy':
        music_file = "path/to/happy_music.mp3"
    elif emotion == 'sad':
        music_file = "path/to/sad_music.mp3"
    elif emotion == 'excited':
        music_file = "path/to/exciting_music.mp3"
    elif emotion == 'scared':
        music_file = "path/to/scary_music.mp3"
    else:
        music_file = "path/to/neutral_music.mp3"

    return pydub.AudioSegment.from_file(music_file)


# Main function to drive the application
def main():
    print("Welcome to the AI-Powered Audio Storyteller!")

    # Record user audio
    audio_file = record_audio()

    # Detect user emotion
    emotion = detect_emotion(audio_file)
    print(f"Detected emotion: {emotion}")

    # Generate story based on emotion
    story = generate_story(emotion)
    print("Generated story:")
    print(story)

    # Play background music
    music = add_background_music(emotion)
    play(music)

    # Narrate the story
    narrate_story(story)

    print("Thank you for using the AI-Powered Audio Storyteller!")


if __name__ == "__main__":
    main()
