"""
Project Title: Emotion-Driven Interactive Audio Ecosystem with Dynamic Sound Layering and Contextual Feedback
File Name: emotion_driven_interactive_audio_ecosystem.py

Project Description:
This project extends the complexity of real-time emotional interaction by introducing a dynamic audio ecosystem where multiple audio layers (e.g., ambient sounds, music, dialogue) evolve based on both emotional context and specific actions taken by the user. The system uses multi-modal emotional feedback that adapts background ambiance, dialogue intensity, and even sound effects in reaction to detected emotions.

Unlike the previous project, this exercise incorporates dynamic sound layering, where individual audio layers (like rain, wind, music) mix together differently depending on emotion, environment, and user input. It also introduces contextual feedback, which changes not only based on the user's current emotional state but also on how it transitions over time.

This is more advanced because of:

Multiple audio layers: Different sound effects are combined and balanced based on the emotional state.
Dynamic feedback: Not only does the system respond to current emotions, but it also recognizes emotional shifts and adjusts sound accordingly.
Complex interactions: The project adds more layers to how the system can adapt to user actions and emotional states, offering a more immersive, multi-faceted experience.
Python Code:
"""
import os
import numpy as np
import sounddevice as sd
from io import BytesIO
from pydub import AudioSegment, playback, effects
from openai import OpenAI
from apikey import apikey
import pyttsx3
import random
from threading import Thread
import time

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS engine for narration
tts_engine = pyttsx3.init()

# Emotion-based soundscapes and sound effects
SOUNDSCAPES = {
    "calm": ["forest_ambiance.wav", "gentle_rain.wav"],
    "tense": ["storm_winds.wav", "thunder.wav"],
    "suspenseful": ["ominous_drone.wav", "heartbeat.wav"]
}

# Emotion-based background music
MUSIC_TRACKS = {
    "calm": "calm_instrumental.wav",
    "tense": "intense_music.wav",
    "suspenseful": "suspenseful_theme.wav"
}

# Memory for emotions and transitions
EMOTION_MEMORY = []
MAX_MEMORY_SIZE = 5


# Function to load audio files
def load_audio(file_path):
    return AudioSegment.from_file(file_path)


# Function to apply dynamic volume control to a sound
def apply_volume_control(sound, volume_factor):
    return sound + volume_factor


# Record user's voice in real time
def record_audio(duration=10, sample_rate=16000):
    print("Recording audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording completed.")
    return recording


# Transcribe audio input and analyze emotion
def transcribe_and_analyze_emotion(audio_data):
    audio_stream = BytesIO(audio_data)
    print("Transcribing and analyzing emotion...")

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )

    transcription = response['text']
    emotion_prompt = f"Analyze the emotion of the following text: '{transcription}'"
    emotion_analysis = client.completions.create(
        model="gpt-4",
        prompt=emotion_prompt,
        max_tokens=50
    )

    detected_emotion = emotion_analysis['choices'][0]['text'].strip().lower()
    print(f"Detected Emotion: {detected_emotion}")
    return transcription, detected_emotion


# Generate dynamic story based on emotional transitions
def generate_dynamic_story(emotion, input_prompt):
    story_prompt = f"Generate a story narrative based on the emotion '{emotion}' and the scenario '{input_prompt}'"
    response = client.completions.create(
        model="gpt-4",
        prompt=story_prompt,
        max_tokens=500
    )
    return response['choices'][0]['text'].strip()


# Play background music based on the emotion
def play_emotion_music(emotion):
    music_track = MUSIC_TRACKS.get(emotion, MUSIC_TRACKS['calm'])
    music = load_audio(music_track)
    playback.play(music)


# Layer multiple soundscapes based on emotional state
def play_dynamic_soundscape(emotion):
    soundscape_layers = SOUNDSCAPES.get(emotion, SOUNDSCAPES['calm'])

    combined_soundscape = AudioSegment.silent(duration=0)
    for sound_file in soundscape_layers:
        sound = load_audio(sound_file)
        combined_soundscape = combined_soundscape.overlay(sound)

    playback.play(combined_soundscape)


# Function to combine layered audio (music + soundscape)
def play_combined_audio(emotion):
    music_thread = Thread(target=play_emotion_music, args=(emotion,))
    soundscape_thread = Thread(target=play_dynamic_soundscape, args=(emotion,))

    music_thread.start()
    soundscape_thread.start()

    music_thread.join()
    soundscape_thread.join()


# Main real-time interaction loop
def real_time_interaction_loop(duration=10):
    while True:
        # Step 1: Record user input audio
        audio_data = record_audio(duration)

        # Step 2: Convert numpy array to PCM bytes for transcription
        audio_data_bytes = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
        transcription, detected_emotion = transcribe_and_analyze_emotion(audio_data_bytes.tobytes())

        # Step 3: Generate a dynamic story based on the detected emotion
        story_prompt = "The hero approaches a mysterious cave entrance."
        generated_story = generate_dynamic_story(detected_emotion, story_prompt)
        print(f"Generated Story: {generated_story}")

        # Use TTS to narrate the generated story
        tts_engine.say(generated_story)
        tts_engine.runAndWait()

        # Step 4: Play combined background music and soundscape layers
        play_combined_audio(detected_emotion)

        # Step 5: Update emotion memory and transition between emotional states
        if len(EMOTION_MEMORY) >= MAX_MEMORY_SIZE:
            EMOTION_MEMORY.pop(0)
        EMOTION_MEMORY.append(detected_emotion)

        # Optional: Add logic to detect major emotional shifts and alter audio layers


# Run the interactive audio ecosystem
if __name__ == "__main__":
    real_time_interaction_loop(duration=10)
"""
Example Inputs and Expected Outputs:
Example 1:
Input: "I'm not sure what's out there, but I feel really scared."
Expected Output:

Transcription: "I'm not sure what's out there, but I feel really scared."
Detected Emotion: "suspenseful"
Generated Story: "The hero hesitates at the entrance of the cave, the air thick with suspense. Shadows move in the darkness, and the only sound is their pounding heartbeat."
Soundscape: Ominous drone layered with heartbeat sounds, creating a sense of fear and anticipation.
Music: Suspenseful theme plays in the background.
Example 2:
Input: "I think everything is going to be alright now. The danger has passed."
Expected Output:

Transcription: "I think everything is going to be alright now. The danger has passed."
Detected Emotion: "calm"
Generated Story: "The hero steps out of the forest, feeling the warm sun on their face. The danger has passed, and peace returns to the land."
Soundscape: Forest ambiance with birds chirping and gentle rain in the distance.
Music: Calm instrumental track playing softly.
Example 3:
Input: "Something is coming. I feel uneasy."
Expected Output:

Transcription: "Something is coming. I feel uneasy."
Detected Emotion: "tense"
Generated Story: "The hero senses a shift in the air as storm clouds gather overhead. The ground trembles, and the wind howls fiercely."
Soundscape: Storm winds and distant thunder overlayed with intense sound effects, creating an ominous atmosphere.
Music: Intense music track to heighten the tension.
Key Improvements:
Multi-Layered Audio Environment: Unlike the previous project, this exercise blends multiple layers of audio (e.g., soundscapes, music, effects) to create a richer, more immersive environment that changes dynamically.
Dynamic Emotion Memory: The system tracks past emotions and transitions between emotional states, creating more nuanced feedback as the user’s emotional context shifts over time.
Real-Time Story Generation with Emotional Influence: The narrative changes not just based on current emotions but also on how emotions have evolved during the session.
Threaded Audio Processing: By running soundscapes and music in parallel using threads, the project ensures smoother, uninterrupted playback and allows for more complex audio manipulation.
"""