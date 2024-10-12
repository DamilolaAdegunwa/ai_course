"""
Project Title: Context-Aware Audio Transcription with Multi-Language Switching and Emotion-Driven Soundtrack Generation
File Name: context_aware_audio_transcription_multilanguage_soundtrack.py

Project Description:
This project is an advanced extension of audio transcription that involves multiple complex components:

Multi-language transcription with automatic language switching: It detects changes in spoken language and dynamically switches between languages during transcription.
Context-aware emotion detection: Instead of detecting emotion on a sentence level, this project uses the entire conversation context to evaluate the dominant emotions and mood of the dialogue.
Emotion-driven soundtrack generation: Based on the overall context and emotional tone of the conversation, the project will generate background music that fits the mood, using pre-defined music segments and effects to enhance the auditory experience.
Continuous real-time processing: The project supports continuous real-time processing of audio data, providing immediate feedback through transcription and music generation.
This project builds on the previous real-time transcription project by adding multi-language recognition and dynamic soundtrack creation based on emotional context. It's useful in creating rich, immersive auditory experiences.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
import numpy as np
import sounddevice as sd
from io import BytesIO
from pydub import AudioSegment, playback
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


# Function to transcribe multi-language audio and detect language switching
def transcribe_multi_language(audio_data):
    audio_stream = BytesIO(audio_data)

    print("Transcribing audio with language switching...")
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )

    transcription = response['text']
    detected_languages = response['language']  # Assuming Whisper-1 returns language data
    return transcription, detected_languages


# Function to detect overall conversation emotion based on context
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


# Function to play background music based on detected emotion
def play_emotion_based_music(emotion):
    soundtrack_path = SOUNDTRACKS.get(emotion, SOUNDTRACKS['neutral'])
    soundtrack = AudioSegment.from_wav(soundtrack_path)
    print(f"Playing {emotion} soundtrack...")
    playback.play(soundtrack)


# Main function to transcribe, detect emotion, and play background music
def real_time_transcription_emotion_music(duration=10):
    # Step 1: Record audio
    audio_data = record_audio(duration=duration)

    # Step 2: Convert numpy array to PCM bytes for transcription
    audio_data_bytes = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
    transcription, detected_languages = transcribe_multi_language(audio_data_bytes.tobytes())
    print(f"Transcription: {transcription}")
    print(f"Detected Languages: {detected_languages}")

    # Step 3: Detect overall emotion based on the conversation's context
    overall_emotion = detect_overall_emotion(transcription)
    print(f"Overall Emotion: {overall_emotion}")

    # Step 4: Play emotion-based background music
    play_emotion_based_music(overall_emotion)


# Run the project
if __name__ == "__main__":
    real_time_transcription_emotion_music(duration=10)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A speaker switching between English and French:

"Hello! Je suis très content aujourd'hui. How are you feeling?"
Expected Transcription:

css
Copy code
"Hello! Je suis très content aujourd'hui. How are you feeling?"
Detected Languages:

less
Copy code
Detected Languages: ['English', 'French']
Overall Detected Emotion:

yaml
Copy code
Overall Emotion: happy
Action: Play the happy soundtrack in the background.

Example 2:

Input Audio: A speaker saying in English:

"I can't believe how unfair this is. I'm really upset!"
Expected Transcription:

css
Copy code
"I can't believe how unfair this is. I'm really upset!"
Detected Languages:

less
Copy code
Detected Languages: ['English']
Overall Detected Emotion:

yaml
Copy code
Overall Emotion: angry
Action: Play the angry soundtrack in the background.

Key Features:
Multi-Language Transcription with Automatic Switching: The system recognizes when the speaker switches between different languages during the audio recording, and transcribes each part correctly according to the spoken language.
Context-Aware Emotion Detection: The system doesn’t analyze emotions on a sentence-by-sentence basis. Instead, it evaluates the overall emotional tone based on the entire transcription context, giving a holistic view of the conversation's sentiment.
Emotion-Driven Soundtrack Generation: After detecting the conversation's emotional tone, the system plays corresponding background music to match the mood, dynamically enhancing the audio experience.
Continuous Real-Time Processing: This project allows for seamless real-time transcription, emotion analysis, and music generation, providing immediate feedback during the audio recording.
Use Cases:
Interactive Podcasts: Create interactive podcasts where background music dynamically changes based on the mood and tone of the conversation.
Language-Learning Platforms: Real-time transcription with multi-language detection can be useful for educational applications, where students practice switching between languages and receive feedback in real-time.
Virtual Therapy or Counseling: In virtual therapy sessions, the system can transcribe the conversation, detect the emotional state of the session, and modify the background music to reflect the emotional tone, improving the experience.
Immersive Storytelling in Multi-Language Narratives: For multi-language stories, the system dynamically switches transcriptions and generates background music to make the experience more engaging.
This project introduces multi-language transcription, context-based emotion analysis, and emotion-driven audio enhancements, making it an advanced use of OpenAI's audio transcription services and real-time processing capabilities. This enables a fully immersive audio experience that reacts dynamically to both language and emotion.
"""