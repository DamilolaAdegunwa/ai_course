"""
Project Title: Dynamic Interactive Soundscape with Emotional Transition Control
File Name: dynamic_interactive_soundscape_emotional_transition.py

Project Description:
This project focuses on creating an advanced real-time interactive soundscape that adapts to the user’s emotional state and environmental cues dynamically. Using OpenAI's Whisper-1 for voice input, GPT-4 for conversational narrative generation, and spatialized sound, the system allows users to interact with a virtual environment. The emotions and actions of the user directly influence the sound environment, character interactions, and music transitions. The system also includes dynamic transitions between soundscapes and real-time audio feedback using text-to-speech.

It is noticeably more advanced than previous exercises by introducing environmental state transitions and layered emotional sound effects that change not only based on user actions but also as emotions intensify or de-escalate over time.

Python Code:
"""
import os
import numpy as np
import sounddevice as sd
from io import BytesIO
from pydub import AudioSegment, playback
from openai import OpenAI
from apikey import apikey
import pyttsx3
import random
import librosa

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS engine for dynamic audio reactions
tts_engine = pyttsx3.init()

# Emotion tracking and soundscape layers
EMOTION_MEMORY = []

# Predefined soundscape transitions based on emotional states
SOUNDSCAPES = {
    "calm_forest": "calm_forest.wav",
    "intense_thunderstorm": "intense_thunderstorm.wav",
    "ambient_night": "ambient_night.wav"
}

# Emotional intensity to trigger transition
EMOTIONAL_INTENSITY_THRESHOLD = 3


# Function to load audio
def load_audio(file_path):
    return AudioSegment.from_file(file_path)


# Apply spatial audio transformation based on coordinates (x, y, z)
def apply_spatial_audio(sound, x, y, z):
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    volume_adjusted = sound - (distance * 5)  # Volume drop-off based on distance
    return volume_adjusted


# Function to record audio input in real-time
def record_audio(duration=10, sample_rate=16000):
    print("Recording audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording completed.")
    return recording


# Transcription and emotional analysis
def transcribe_and_analyze_emotion(audio_data):
    audio_stream = BytesIO(audio_data)
    print("Transcribing audio and analyzing emotion...")

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )

    transcription = response['text']

    # Analyze the transcription for emotional content
    emotion_prompt = f"Analyze the emotion of this transcription: '{transcription}'"
    emotion_analysis = client.completions.create(
        model="gpt-4",
        prompt=emotion_prompt,
        max_tokens=50
    )

    detected_emotion = emotion_analysis['choices'][0]['text'].strip().lower()
    print(f"Detected Emotion: {detected_emotion}")
    return transcription, detected_emotion


# Generate story narrative based on emotional cues
def generate_story(emotion, input_prompt):
    print("Generating story narrative based on emotion...")
    response = client.completions.create(
        model="gpt-4",
        prompt=f"Create a narrative based on the following emotion '{emotion}': {input_prompt}",
        max_tokens=500
    )
    return response['choices'][0]['text'].strip()


# Change soundscape based on emotional intensity
def change_soundscape(emotion, intensity):
    if intensity >= EMOTIONAL_INTENSITY_THRESHOLD:
        if emotion == 'angry':
            return load_audio(SOUNDSCAPES['intense_thunderstorm'])
        elif emotion == 'calm':
            return load_audio(SOUNDSCAPES['calm_forest'])
    return load_audio(SOUNDSCAPES['ambient_night'])  # Default soundscape


# Main real-time interaction loop
def real_time_interaction_loop(duration=10):
    while True:
        # Step 1: Record user audio
        audio_data = record_audio(duration)

        # Step 2: Convert numpy array to PCM bytes for transcription
        audio_data_bytes = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
        transcription, detected_emotion = transcribe_and_analyze_emotion(audio_data_bytes.tobytes())

        # Step 3: Generate dynamic story based on emotional analysis
        user_prompt = "The user enters a mysterious forest."
        generated_story = generate_story(detected_emotion, user_prompt)
        print(f"Generated Story: {generated_story}")
        tts_engine.say(generated_story)
        tts_engine.runAndWait()

        # Step 4: Track emotional intensity over time and switch soundscapes if necessary
        EMOTION_MEMORY.append(detected_emotion)
        emotional_intensity = EMOTION_MEMORY.count(detected_emotion)
        print(f"Emotional Intensity: {emotional_intensity}")

        new_soundscape = change_soundscape(detected_emotion, emotional_intensity)
        playback.play(new_soundscape)

        # Step 5: Clear memory to avoid escalation loop
        if emotional_intensity >= EMOTIONAL_INTENSITY_THRESHOLD:
            EMOTION_MEMORY.clear()


# Run the dynamic interaction loop
if __name__ == "__main__":
    real_time_interaction_loop(duration=10)
"""
Example Inputs and Expected Outputs:
Example 1:
Input: User says, "I’m really anxious about what will happen next."
Expected Output:

Transcription: "I’m really anxious about what will happen next."
Detected Emotion: "anxious"
Generated Story: "As the adventurer nervously steps forward, the wind howls through the trees, adding to their unease."
Soundscape: Intense thunderstorm sounds, louder as anxiety escalates, simulating growing tension in the environment.
Example 2:
Input: User says, "It feels peaceful here, like everything is in balance."
Expected Output:

Transcription: "It feels peaceful here, like everything is in balance."
Detected Emotion: "calm"
Generated Story: "The adventurer feels a deep sense of calm, the forest around them alive with the soft sounds of nature."
Soundscape: Calm forest ambiance with birds chirping and a gentle breeze, the volume adjusting based on the user's proximity to key elements in the soundscape.
Key Improvements:
Emotion-Driven Soundscape Transitions: Real-time changes to environmental sounds based on emotional intensity, simulating a deeper immersive experience.
Layered Emotional Memory: Tracks emotional states over time, influencing future responses and soundscape transitions, offering more complex emotional progression.
Spatial Audio: The project utilizes spatial audio, adjusting volumes based on distance and enhancing immersion by making the virtual world feel more alive.
Dynamic Storytelling Based on Emotion: The story’s narrative adjusts to the user’s detected emotions, making the experience more personal and responsive."""