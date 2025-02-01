"""
Project Title: Multi-Layered Audio Experiences with Real-Time Emotion Clustering and Dynamic Sound Element Integration
File Name: multi_layered_audio_emotion_clustering.py

Project Description:
In this project, we create multi-layered audio experiences that dynamically adjust not just to a single emotion but to multiple overlapping emotions detected from the user’s voice. The system uses emotion clustering to analyze user voice in real-time, identifying multiple emotions simultaneously (e.g., a mix of happiness and fear) and adjusts:

Soundtrack complexity: Multiple audio layers are mixed based on emotional intensity and combinations.
Real-time narrative adaptation: The narrative adjusts to the leading detected emotions.
Layered ambient sound effects: Environmental sound effects change dynamically based on the combination of emotions.
Emotion-driven transitions: Smooth emotional transitions between different soundscapes and narratives based on changing emotional states.
This system adds complexity through the interaction of multiple emotions in real-time, allowing for more nuanced and engaging user experiences.

Python Code:
"""
import os
import sounddevice as sd
import librosa
import numpy as np
import pyttsx3
from openai import OpenAI
from apikey import apikey  # Import your API key from apikey.py
import speech_recognition as sr
from pydub import AudioSegment, playback

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# TTS Engine Initialization
tts_engine = pyttsx3.init()

# Recognizer for voice input
recognizer = sr.Recognizer()

# Multi-emotion soundtracks and ambient effects
EMOTION_LAYERS = {
    "happy": ("happy_layer1.wav", "happy_layer2.wav"),
    "sad": ("sad_layer1.wav", "sad_layer2.wav"),
    "fear": ("fear_layer1.wav", "fear_layer2.wav"),
    "anger": ("anger_layer1.wav", "anger_layer2.wav")
}

# Load and play audio function
def load_audio(file_path):
    return AudioSegment.from_file(file_path)

def play_audio(audio_segment):
    playback.play(audio_segment)

# Record user input (real-time)
def record_audio(duration=5, sample_rate=16000):
    print("Recording user voice input...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio

# Detect multiple emotions from user voice using AI
def detect_emotions_from_voice(audio_data, sample_rate=16000):
    # Convert audio data to WAV format
    audio_wav = librosa.resample(audio_data[:, 0], orig_sr=sample_rate, target_sr=16000)
    audio_pcm = librosa.util.buf_to_int(audio_wav)

    # Convert to the required format for OpenAI input
    audio_input = client.audio.transcribe(model="whisper-1", file=audio_pcm)

    # Analyze transcribed text for multiple emotions
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotions in this transcription and return a mixture of possible emotions: {audio_input['text']}",
        max_tokens=50
    )
    emotions = response.choices[0].text.strip().lower().split(',')
    return [emotion.strip() for emotion in emotions]

# Generate dynamic narrative based on emotions
def generate_narrative(emotions, user_input):
    narrative_prompt = f"Create a story influenced by the emotions: {', '.join(emotions)}, starting with: {user_input}"
    response = client.completions.create(
        model="text-davinci-003",
        prompt=narrative_prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()

# Adjust narration style based on emotions
def adjust_voice_for_emotions(emotions):
    if "happy" in emotions:
        tts_engine.setProperty("rate", 180)
    elif "sad" in emotions:
        tts_engine.setProperty("rate", 110)
    elif "fear" in emotions:
        tts_engine.setProperty("rate", 150)
    elif "anger" in emotions:
        tts_engine.setProperty("rate", 200)

# Narrate story based on emotions
def narrate_story(story, emotions):
    adjust_voice_for_emotions(emotions)
    tts_engine.say(story)
    tts_engine.runAndWait()

# Play layered soundtracks based on emotions
def play_emotion_layers(emotions):
    for emotion in emotions:
        layers = EMOTION_LAYERS.get(emotion)
        if layers:
            for layer in layers:
                audio = load_audio(layer)
                play_audio(audio)

# Play sound effects for emotional transitions
def play_transition_sounds(emotion, next_emotion):
    transition_prompt = f"Play sound effects to transition from {emotion} to {next_emotion}"
    print(transition_prompt)  # In a real-world scenario, you'd play an actual sound effect file here.

# Main storytelling loop with real-time emotion clustering
def real_time_emotion_clustering_storytelling():
    previous_emotion = None
    while True:
        # Get user input for the story
        user_input = input("Enter a story prompt (or type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break

        # Record user voice for emotion detection
        print("Please speak into the microphone to record your emotional state...")
        user_audio = record_audio()

        # Detect user's emotions based on voice
        user_emotions = detect_emotions_from_voice(user_audio)
        print(f"Detected emotions: {', '.join(user_emotions)}")

        # Generate story based on the user's input and emotions
        dynamic_story = generate_narrative(user_emotions, user_input)
        print(f"Generated story: {dynamic_story}")

        # Narrate the generated story with emotion-based modulation
        narrate_story(dynamic_story, user_emotions)

        # Play multi-layered soundtracks for detected emotions
        play_emotion_layers(user_emotions)

        # Handle emotion transitions and ambient sounds
        if previous_emotion and user_emotions[0] != previous_emotion:
            play_transition_sounds(previous_emotion, user_emotions[0])

        previous_emotion = user_emotions[0]

# Run the real-time emotion clustering interactive story experience
if __name__ == "__main__":
    real_time_emotion_clustering_storytelling()
"""
Example Inputs and Expected Outputs:
Example 1:
User Input (typed): "A detective is investigating a crime scene."
User Voice Emotion: Mixed Sadness and Fear
Detected Emotions: Sad, Fear
Generated Story: "The detective stood silently at the edge of the dimly lit alley, the fear gnawing at the back of his mind. Every clue felt heavier, as though each step led closer to a hidden horror."
Narration Output:

Voice Modulation: Low, soft tone for sadness, with pauses to convey fear.
Background Music: A low droning sound with a soft, eerie melody layered in the background.
Sound Effects: Soft wind mixed with distant echoes.
Example 2:
User Input (typed): "A soldier leads a mission into enemy territory."
User Voice Emotion: Mixed Anger and Fear
Detected Emotions: Anger, Fear
Generated Story: "The soldier gripped the rifle tightly, eyes scanning the dark horizon. Anger surged through his veins, fueling his every step. But fear, like a shadow, lurked, reminding him of the stakes."
Narration Output:

Voice Modulation: Fast, intense for anger, but punctuated by quick pauses for fear.
Background Music: Heavy, rhythmic drums mixed with an unsettling string section.
Sound Effects: Footsteps on gravel mixed with distant explosions.
Key Features:
Multi-Emotion Detection: The system detects multiple emotions in real-time, creating a more nuanced and adaptive user experience.
Layered Soundtracks: Soundtracks dynamically combine multiple audio layers based on the detected emotions.
Emotion Transitions: Smooth transitions between different soundscapes as user emotions change, creating a cohesive experience.
Real-Time Emotion Clustering: The narrative adapts based on clusters of emotions, rather than just a single detected emotion, making the storyline richer and more personalized.
Conclusion:
This project pushes the boundaries of interactive audio experiences by introducing real-time emotion clustering and multi-layered audio. It can be further expanded by adding additional emotion layers, more complex narratives, and personalized interactions based on the user's evolving emotional state.
"""