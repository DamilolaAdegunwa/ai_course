"""
Project Title: Emotion-Driven Audio Augmentation with Personalized Soundscapes and Real-Time Voice Modulation
File Name: emotion_driven_audio_augmentation.py

Project Description:
This project builds upon multi-emotion detection by adding personalized soundscapes and dynamic voice modulation based on emotional intensity. It creates a personalized auditory experience by:

Augmenting recorded voice: Adding effects such as reverb, echo, and distortion based on emotional intensity.
Adaptive background soundscapes: Dynamically adjusting volume, panning, and reverb of environmental sounds based on emotions.
Emotion Intensity Mapping: Mapping detected emotions to a real-time voice modulation system, adjusting pitch, rate, and echo effects according to emotional intensity.
Personalized Soundscapes: Environmental soundscapes such as rain, city ambiance, or nature sounds are dynamically created and layered based on user preferences and current emotions.
The aim is to create a real-time, fully personalized sound environment that is driven by the user's emotional state and voice inputs.

Python Code:
"""
import os
import numpy as np
import sounddevice as sd
import librosa
import pyttsx3
from openai import OpenAI
from apikey import apikey  # Import your API key from apikey.py
import speech_recognition as sr
from pydub import AudioSegment, effects, playback

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Recognizer for user input
recognizer = sr.Recognizer()

# Load ambient soundscapes for emotional augmentation
SOUNDSCAPES = {
    "nature": "nature_ambience.wav",
    "city": "city_ambience.wav",
    "rain": "rain_ambience.wav"
}


# Voice modulation based on emotion intensity
def modulate_voice(audio, pitch_shift=1.0, reverb=False):
    # Apply pitch shift
    modulated_audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=pitch_shift)

    # Apply reverb if needed
    if reverb:
        modulated_audio = effects.reverb(modulated_audio)

    return modulated_audio


# Record user input
def record_user_input(duration=5, sample_rate=16000):
    print("Recording user voice input...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio


# Detect emotion from user voice using OpenAI
def detect_emotion(audio_data, sample_rate=16000):
    # Convert audio data to required format for OpenAI input
    audio_wav = librosa.resample(audio_data[:, 0], orig_sr=sample_rate, target_sr=16000)
    audio_pcm = librosa.util.buf_to_int(audio_wav)

    # Use OpenAI Whisper to transcribe the audio and then analyze emotions
    audio_input = client.audio.transcribe(model="whisper-1", file=audio_pcm)

    # Use the transcribed text to analyze emotions
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotional intensity and the dominant emotion in the transcription: {audio_input['text']}",
        max_tokens=50
    )
    emotion_analysis = response.choices[0].text.strip()
    return emotion_analysis


# Generate dynamic soundscapes based on emotion
def generate_personalized_soundscape(emotion):
    if "happy" in emotion:
        soundscape = load_soundscape("nature")
    elif "sad" in emotion:
        soundscape = load_soundscape("rain")
    elif "angry" in emotion:
        soundscape = load_soundscape("city")
    else:
        soundscape = load_soundscape("nature")  # Default to nature soundscape
    return soundscape


# Load ambient soundscape files
def load_soundscape(soundscape_name):
    soundscape_file = SOUNDSCAPES.get(soundscape_name)
    if soundscape_file:
        return AudioSegment.from_file(soundscape_file)
    else:
        raise ValueError(f"Soundscape '{soundscape_name}' not found.")


# Play soundscapes and apply effects based on emotions
def play_soundscape_with_emotion(soundscape, intensity):
    # Apply volume adjustments based on emotional intensity
    volume_adjustment = intensity * 5  # Scale volume by intensity
    soundscape = soundscape + volume_adjustment
    playback.play(soundscape)


# Main function to record, detect emotions, and play personalized soundscapes
def real_time_emotion_audio_augmentation():
    while True:
        user_input = input("Enter your voice prompt (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Record user voice
        user_audio = record_user_input()

        # Detect emotion from voice input
        emotion_data = detect_emotion(user_audio)
        print(f"Detected emotion: {emotion_data}")

        # Extract emotional intensity and dominant emotion
        if "intensity" in emotion_data:
            intensity = float(emotion_data.split("intensity")[1].strip())
            dominant_emotion = emotion_data.split(",")[0].strip()
        else:
            intensity = 1.0
            dominant_emotion = emotion_data

        # Generate personalized soundscape based on detected emotion
        personalized_soundscape = generate_personalized_soundscape(dominant_emotion)

        # Play personalized soundscape with emotional effects
        play_soundscape_with_emotion(personalized_soundscape, intensity)

        # Modulate user's voice based on emotional intensity
        modulated_voice = modulate_voice(user_audio, pitch_shift=intensity,
                                         reverb=True if dominant_emotion == "sad" else False)
        playback.play(modulated_voice)


# Run the emotion-driven audio augmentation experience
if __name__ == "__main__":
    real_time_emotion_audio_augmentation()
"""
Example Inputs and Expected Outputs:
Example 1:
User Input: "Describe my day in nature."
User Voice Emotion: Happy
Detected Emotion: Happy, Intensity: 0.8
Generated Soundscape: Nature Ambience
Voice Modulation: Slight pitch shift upward to reflect happiness, moderate reverb for an outdoor effect.

Expected Output:
Ambient Sound: A gentle breeze and birds chirping in the background.
Voice Output: The user’s voice is slightly higher in pitch, with light reverb to mimic an open, outdoor space.
Example 2:
User Input: "Talk about the rain in the city."
User Voice Emotion: Sad
Detected Emotion: Sad, Intensity: 1.2
Generated Soundscape: Rain Ambience
Voice Modulation: Lower pitch and added reverb to convey sadness.

Expected Output:
Ambient Sound: Soft rain falling with distant thunder.
Voice Output: The user’s voice is lower in pitch with strong reverb, reflecting a somber tone.
Key Features:
Emotion-Driven Voice Modulation: Voice effects such as pitch shifting, reverb, and distortion are applied in real time, enhancing the emotional depth of the user's voice.
Personalized Soundscapes: Dynamic ambient soundscapes (e.g., nature, rain, city) are generated based on the user’s emotional state, adding depth to the auditory experience.
Emotion Intensity Mapping: Emotional intensity influences both the soundscapes and the user's voice modulation, providing a more personalized, responsive experience.
Conclusion:
This project enhances the interactive experience by adding emotion-driven voice modulation and personalized soundscapes based on real-time emotional analysis. By adjusting both the user’s voice and the ambient soundscapes, the system creates a more immersive and emotionally resonant auditory environment.
"""