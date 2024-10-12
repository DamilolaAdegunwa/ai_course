"""
Project Title: Audio Transcription with Sentiment-Driven Dynamic Music Generation
File Name: audio_transcription_with_sentiment_music_generation.py

Project Description:
In this more advanced OpenAI audio project, we will create a system that:

Transcribes speech from an audio file.
Performs sentiment analysis on the transcribed text.
Generates music dynamically based on the detected sentiment of the speaker, with different styles (e.g., calm, upbeat, melancholic, etc.) based on the emotional tone of the conversation.
Plays the dynamically generated music to the user to match the conversation mood in real-time.
This project adds several advanced concepts, such as real-time sentiment-driven music generation, a significant leap from the previous project which focused on emotion detection and dialogue generation.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
import wave
import numpy as np
from io import BytesIO
import pyaudio

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Load sentiment and music generation model info
sentiment_labels = ['happy', 'sad', 'angry', 'neutral', 'relaxed', 'excited']


# Function to transcribe the audio file
def transcribe_audio(file_path):
    audio_file = open(file_path, "rb")

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )

    transcription = response['text']
    return transcription


# Function to analyze sentiment of the transcribed text
def analyze_sentiment(transcription):
    sentiment_prompt = f"Analyze the sentiment of this text: '{transcription}' and indicate if it's happy, sad, angry, relaxed, excited, or neutral."

    sentiment_response = client.completions.create(
        model="text-davinci-003",
        prompt=sentiment_prompt,
        max_tokens=50
    )

    sentiment = sentiment_response['choices'][0]['text'].strip().lower()
    if sentiment not in sentiment_labels:
        sentiment = 'neutral'

    return sentiment


# Function to generate music based on sentiment
def generate_music(sentiment):
    music_prompt = f"Generate a music description that suits the '{sentiment}' sentiment. Describe the instruments, tempo, and feel."

    music_description_response = client.completions.create(
        model="text-davinci-003",
        prompt=music_prompt,
        max_tokens=100
    )

    music_description = music_description_response['choices'][0]['text'].strip()
    print(f"Generated music description: {music_description}")

    # Create basic wave sound based on sentiment (example: change frequency/tempo)
    frequency = 440  # Base frequency for music tone
    duration = 2.0  # Music duration in seconds

    if sentiment == 'happy' or sentiment == 'excited':
        frequency = 660  # Higher frequency for upbeat tones
    elif sentiment == 'sad' or sentiment == 'melancholic':
        frequency = 220  # Lower frequency for sadder tones
    elif sentiment == 'angry':
        frequency = 880  # Very high-pitched for intense feeling
    elif sentiment == 'relaxed':
        frequency = 330  # Calm, relaxing frequency

    return create_tone(frequency, duration)


# Function to create a basic sound wave to represent music
def create_tone(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Ensure data is in 16-bit format for playback
    audio_data = np.int16(tone * 32767)
    audio_data = BytesIO(audio_data.tobytes())

    return audio_data


# Function to play generated music
def play_generated_music(audio_data, sample_rate=44100):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True)

    stream.write(audio_data.getvalue())

    stream.stop_stream()
    stream.close()
    p.terminate()


# Main function to run transcription and dynamic music generation
def transcribe_and_generate_music(file_path):
    # Step 1: Transcribe the audio
    transcription = transcribe_audio(file_path)
    print(f"Transcription: {transcription}")

    # Step 2: Analyze the sentiment of the transcription
    sentiment = analyze_sentiment(transcription)
    print(f"Detected sentiment: {sentiment}")

    # Step 3: Generate music based on the sentiment
    generated_music = generate_music(sentiment)

    # Step 4: Play the generated music
    print(f"Playing music for sentiment: {sentiment}")
    play_generated_music(generated_music)


# Run the program with the given audio file
if __name__ == "__main__":
    file_path = r"C:\path_to_audio\sample_audio_file.mp3"
    transcribe_and_generate_music(file_path)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio (Speaker 1): "I'm really excited! We just landed the biggest deal for our company."

Expected Transcription: "I'm really excited! We just landed the biggest deal for our company."

Expected Sentiment: "excited"

Generated Music: Fast-paced, high-pitched music with upbeat tones.

Action: The system plays upbeat music that matches the "excited" sentiment.

Example 2:

Input Audio (Speaker 1): "It's been a rough day. I feel so sad."

Expected Transcription: "It's been a rough day. I feel so sad."

Expected Sentiment: "sad"

Generated Music: Slow, deep, melancholic music with low tones.

Action: The system plays slow, calming music matching the "sad" sentiment.

Key Features:
Real-Time Audio Transcription: Utilizes OpenAI Whisper model to transcribe the speech into text.
Sentiment Analysis: The transcribed text is analyzed to detect the overall emotional tone, whether it’s happy, sad, angry, excited, etc.
Dynamic Music Generation: Based on the sentiment detected, the system generates music using a simple sine wave generator to simulate the mood.
Music Playback: The system immediately plays the music, giving real-time feedback based on the emotional tone of the transcription.
Use Cases:
Interactive Meditation Apps: A meditation app can use this to transcribe what users say and generate music dynamically based on their mood, creating an emotionally personalized experience.
Therapy and Mental Health: In therapy sessions, transcription and sentiment analysis can help therapists understand the emotional state of clients, and the dynamically generated music can set the tone for therapeutic exercises.
Gaming or VR Experiences: Games or VR experiences can use dynamic sentiment-driven music to enhance the emotional depth of player interactions.
This project provides a seamless integration of emotion detection with real-time music generation, creating a powerful system that can adjust the environment's emotional tone based on the speaker’s input.
"""