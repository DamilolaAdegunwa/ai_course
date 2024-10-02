"""
Project Title: Real-Time Language Translation and Speaker Emotion Classification for Live Podcasts
File Name: real_time_language_translation_emotion_classification.py

Project Description:
In this advanced project, we will create a system that:

Streams live audio from a podcast or video feed.
Transcribes the live audio and translates it in real-time into multiple languages using OpenAI's Whisper model.
Detects the speaker's emotion (e.g., happy, angry, sad) and classifies the emotional tone of each sentence.
Outputs both the translated text and the emotion label for every sentence.
Optionally, it generates emotion-enhanced audio output where certain sound effects or background music are added depending on the emotional tone detected.
This project introduces real-time capabilities and multimodal processing by combining both translation and emotion classification into a single system. We will also implement the streaming component, which makes this significantly more complex than previous projects.

Python Code:
"""
import os
import librosa
import sounddevice as sd
import queue
import threading
import numpy as np
from openai import OpenAI
from apikey import apikey
from textblob import TextBlob
from googletrans import Translator  # For multilingual translation
from typing import Tuple, List

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Step 1: Real-Time Audio Streaming
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


def start_audio_stream(sample_rate=16000):
    print("Starting audio stream...")
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate)
    stream.start()


# Step 2: Real-Time Transcription and Emotion Detection
def transcribe_and_analyze_emotion(audio_data: np.ndarray, sample_rate: int) -> Tuple[str, str]:
    print("Transcribing and analyzing emotion...")

    # Save the live audio stream into a temporary file
    temp_file = 'temp_audio.wav'
    librosa.output.write_wav(temp_file, audio_data, sample_rate)

    # Transcribe the audio using OpenAI's Whisper model
    with open(temp_file, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    transcription = response['text']

    # Analyze the emotion of the transcription
    blob = TextBlob(transcription)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        emotion = "Happy"
    elif polarity < -0.2:
        emotion = "Angry"
    else:
        emotion = "Neutral"

    return transcription, emotion


# Step 3: Translate Transcription into Multiple Languages
def translate_text(text: str, target_language: str) -> str:
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


# Step 4: Add Emotion-Based Enhancements to the Audio
def emotion_enhanced_audio(audio_data: np.ndarray, emotion: str) -> np.ndarray:
    print(f"Enhancing audio based on emotion: {emotion}")
    if emotion == "Happy":
        # Slight amplification for happy segments
        enhanced_audio = audio_data * 1.1
    elif emotion == "Angry":
        # Noise reduction for angry segments
        enhanced_audio = librosa.effects.preemphasis(audio_data)
    else:
        # Smoothing for neutral segments
        enhanced_audio = audio_data * 0.9

    return enhanced_audio


# Step 5: Process Live Audio in Real-Time
def process_live_audio(sample_rate=16000, language='es'):
    start_audio_stream(sample_rate)

    while True:
        try:
            audio_chunk = audio_queue.get()
            if audio_chunk is not None:
                # Transcribe and analyze emotion
                transcription, emotion = transcribe_and_analyze_emotion(audio_chunk, sample_rate)

                # Translate transcription
                translation = translate_text(transcription, target_language=language)

                # Enhance audio based on emotion
                enhanced_audio = emotion_enhanced_audio(audio_chunk, emotion)

                # Output transcription, translation, and emotion
                print(f"Original Transcription: {transcription}")
                print(f"Translated Text ({language}): {translation}")
                print(f"Emotion: {emotion}")

                # Play the enhanced audio (Optional)
                sd.play(enhanced_audio, samplerate=sample_rate)

        except Exception as e:
            print(f"Error processing live audio: {e}")


# Main function to run the real-time system
if __name__ == "__main__":
    target_language = 'fr'  # Target translation language (e.g., 'fr' for French)
    process_live_audio(sample_rate=16000, language=target_language)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A live-streamed podcast where two speakers discuss sports, and one speaker is enthusiastic while the other is neutral.

Expected Output:

Original Transcription: "The team played really well today! They dominated the entire game!"
Translated Text (French): "L'équipe a très bien joué aujourd'hui ! Ils ont dominé tout le match !"
Emotion: Happy
Audio Enhancement: Slight amplification for the happy tone in the transcription.
Enhanced Audio Output: Real-time playback of the happy-segment-enhanced audio in the live stream.
Example 2:

Input Audio: A debate between two politicians, where one speaker raises their voice, sounding angry.

Expected Output:

Original Transcription: "You cannot justify these policies! They are harmful to the public!"
Translated Text (Spanish): "¡No puedes justificar estas políticas! ¡Son perjudiciales para el público!"
Emotion: Angry
Audio Enhancement: Apply a noise reduction effect to smooth out the harshness of the angry tone.
Enhanced Audio Output: Real-time playback with noise reduction applied to calm down the aggressive speech.
Key Features:
Real-Time Audio Processing: Streams live audio and processes it in real-time for transcription, translation, emotion classification, and audio enhancement.
Multilingual Translation: Uses Google Translate to translate the live transcription into different languages, providing support for multilingual audiences.
Emotion Detection: Each sentence is analyzed for emotional tone, and labels such as "Happy," "Angry," and "Neutral" are assigned.
Emotion-Based Audio Enhancements: Dynamically modifies the live audio stream, enhancing the playback experience by adjusting the audio based on emotion (e.g., amplification for happy, noise reduction for angry).
Live Audio Playback: Outputs the enhanced audio in real-time, allowing listeners to experience a more refined version of the live podcast or stream.
Use Cases:
Live Podcast Translations: Stream real-time podcast translations for multilingual audiences, with additional emotion-aware audio enhancements for better listening experiences.
International Conferences: Enhance audio for conferences with live translations and dynamic audio effects based on speaker emotions, improving accessibility and engagement.
Live Customer Support: Analyze customer emotions in real-time calls and apply appropriate audio enhancements to improve clarity and tone, while offering translated support for different regions.
Interactive Broadcasting: Broadcasters can integrate this system for real-time analysis and improvement of emotional tones during live debates or discussions.
This project adds significant complexity by incorporating real-time processing, combining both language translation and emotion classification into a live stream. It requires handling continuous audio input, real-time transcription and translation, and dynamic audio modifications, making it a substantial step up from previous projects in terms of functionality and implementation difficulty.
"""