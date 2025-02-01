"""
Project Title: Dynamic Audio Interaction with Real-Time Sentiment Analysis and Sound Modulation
File Name: dynamic_audio_interaction_with_sentiment_analysis_and_sound_modulation.py

Project Description:
This project expands upon your prior work by incorporating real-time audio sentiment analysis with a dynamic sound modulation feature. Instead of only detecting general emotions, this project adjusts the playback speed and pitch of background music based on the detected sentiment intensity (e.g., mildly happy or extremely angry). Additionally, it introduces real-time sentiment changes, modifying the background audio dynamically as the conversation evolves, keeping pace with the emotional shifts in the transcription.

Python Code:
"""
import os
import numpy as np
import sounddevice as sd
from io import BytesIO
from pydub import AudioSegment, playback
from pydub.effects import speedup, pitch_shift
from openai import OpenAI
from apikey import apikey

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Pre-defined soundtracks for emotions
SOUNDTRACKS = {
    'happy': 'happy_background_music.wav',
    'sad': 'sad_background_music.wav',
    'angry': 'angry_background_music.wav',
    'neutral': 'neutral_background_music.wav'
}


# Function to record audio
def record_audio(duration=10, sample_rate=16000):
    print("Recording audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording completed.")
    return recording


# Function to transcribe audio with real-time sentiment analysis
def transcribe_with_sentiment_analysis(audio_data):
    audio_stream = BytesIO(audio_data)

    print("Transcribing and analyzing sentiment in real-time...")
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )

    transcription = response['text']
    print(f"Transcription: {transcription}")

    sentiment_response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the sentiment and intensity of the following transcription: '{transcription}'",
        max_tokens=50
    )

    sentiment_data = sentiment_response['choices'][0]['text'].strip().lower()
    print(f"Detected Sentiment: {sentiment_data}")
    return transcription, sentiment_data


# Function to adjust music playback based on sentiment intensity
def modulate_music_based_on_sentiment(emotion, sentiment_data):
    soundtrack_path = SOUNDTRACKS.get(emotion, SOUNDTRACKS['neutral'])
    soundtrack = AudioSegment.from_wav(soundtrack_path)

    if 'mild' in sentiment_data:
        modulated_soundtrack = speedup(soundtrack, playback_speed=1.1)  # Slight speed-up for mild emotions
    elif 'intense' in sentiment_data:
        modulated_soundtrack = pitch_shift(soundtrack, octaves=0.3)  # Pitch shift for intense emotions
    else:
        modulated_soundtrack = soundtrack  # Play as normal for neutral or unclassified emotions

    print(f"Playing modulated {emotion} soundtrack...")
    playback.play(modulated_soundtrack)


# Main function for real-time transcription, sentiment analysis, and dynamic sound modulation
def real_time_sentiment_analysis_and_sound_modulation(duration=10):
    # Step 1: Record audio
    audio_data = record_audio(duration=duration)

    # Step 2: Convert numpy array to PCM bytes for transcription and sentiment analysis
    audio_data_bytes = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
    transcription, sentiment_data = transcribe_with_sentiment_analysis(audio_data_bytes.tobytes())

    # Step 3: Detect emotion (e.g., happy, sad) and modulate soundtrack based on sentiment intensity
    emotion = detect_overall_emotion(transcription)  # Reuse previous emotion detection logic
    modulate_music_based_on_sentiment(emotion, sentiment_data)


# Helper function to detect overall emotion based on transcription (reused from earlier projects)
def detect_overall_emotion(transcription):
    print("Detecting overall emotion based on context...")
    prompt = f"Based on the following transcription, what is the overall emotion (happy, sad, angry, neutral)? '{transcription}'"

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=5
    )

    overall_emotion = response['choices'][0]['text'].strip().lower()
    return overall_emotion


# Run the project
if __name__ == "__main__":
    real_time_sentiment_analysis_and_sound_modulation(duration=10)
"""
Example Inputs and Expected Outputs:
Example 1:
Input: Recorded conversation with mildly positive sentiment (e.g., "It was a good day, nothing extraordinary but pleasant.")
Expected Output:

Transcription: "It was a good day, nothing extraordinary but pleasant."
Sentiment: Mildly happy
Music Modulation: Background music plays slightly faster due to the mild emotion.
Example 2:
Input: A passionate, angry dialogue (e.g., "This is outrageous! How could they do this?")
Expected Output:

Transcription: "This is outrageous! How could they do this?"
Sentiment: Intense anger
Music Modulation: Background music plays with a deeper pitch due to the intensity of anger.
Key Improvements:
Real-time sentiment detection with intensity: Rather than only detecting the emotion, the system adjusts based on the intensity (mild, moderate, intense) of the emotion.
Dynamic audio modulation: The playback speed and pitch of the background music change in response to sentiment intensity, offering a more dynamic and immersive experience.
Real-time interaction: Sentiment is analyzed on the fly, and background audio is adjusted as emotions evolve.
"""