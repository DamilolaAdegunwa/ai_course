"""
Project Title: Dynamic Audio Narratives with Real-Time Emotion Detection from User Voice and Personalized Adaptive Soundscapes
File Name: dynamic_audio_narratives_with_emotion_detection.py

Project Description:
This project takes interactive audio narratives to the next level by incorporating real-time emotion detection from the user's voice and dynamically personalized adaptive soundscapes. The system will:

Analyze user emotions based on their voice input using AI-driven emotion recognition.
Adjust narration based on the detected emotions, dynamically modifying the narrator’s tone, speed, and background music to fit the user’s emotional state.
Create personalized soundscapes, where ambient sound effects and background music respond both to the storyline and the user's emotions.
Real-time adaptation of the storyline, where narrative progression and interactive elements change based on the user’s current emotional state detected from their voice.
Dynamic Story Alteration: The plotline can be altered based on the emotions detected from the user's voice, making it an ever-changing story.
Python Code:
"""
import os
import sounddevice as sd
import numpy as np
import pyttsx3
from openai import OpenAI
from apikey import apikey  # Import your API key from apikey.py
import librosa
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

# Emotion-based soundscapes
SOUNDTRACKS = {
    "happy": "happy_soundtrack.wav",
    "sad": "sad_soundtrack.wav",
    "angry": "angry_soundtrack.wav",
    "neutral": "calm_soundtrack.wav"
}

# Emotion-based ambient sound effects
AMBIENT_EFFECTS = {
    "happy": "birds_chirping.wav",
    "sad": "soft_rain.wav",
    "angry": "thunder_storm.wav",
    "neutral": "wind_breeze.wav"
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


# Detect emotion from user voice using AI
def detect_emotion_from_voice(audio_data, sample_rate=16000):
    # Convert audio data to WAV format
    audio_wav = librosa.resample(audio_data[:, 0], orig_sr=sample_rate, target_sr=16000)
    audio_pcm = librosa.util.buf_to_int(audio_wav)

    # Convert to the required format for OpenAI input
    audio_input = client.audio.transcribe(model="whisper-1", file=audio_pcm)

    # Analyze transcribed text for emotion
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotion in this transcription: {audio_input['text']}",
        max_tokens=50
    )
    detected_emotion = response.choices[0].text.strip().lower()
    return detected_emotion


# Generate dynamic narrative based on emotion
def generate_narrative(emotion, user_input):
    narrative_prompt = f"Create a story based on a {emotion} emotion, starting with: {user_input}"
    response = client.completions.create(
        model="text-davinci-003",
        prompt=narrative_prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()


# Adjust narration style based on emotion
def adjust_voice_for_emotion(emotion):
    settings = {
        "happy": {"rate": 180, "pitch": 200},
        "sad": {"rate": 110, "pitch": 100},
        "angry": {"rate": 200, "pitch": 250},
        "neutral": {"rate": 150, "pitch": 170}
    }
    tts_engine.setProperty("rate", settings[emotion]["rate"])
    tts_engine.setProperty("pitch", settings[emotion]["pitch"])


# Narrate story based on emotion
def narrate_story(story, emotion):
    adjust_voice_for_emotion(emotion)
    tts_engine.say(story)
    tts_engine.runAndWait()


# Play emotion-based soundtrack and sound effects
def play_emotion_based_audio(emotion):
    soundtrack_file = SOUNDTRACKS.get(emotion, SOUNDTRACKS["neutral"])
    ambient_file = AMBIENT_EFFECTS.get(emotion, AMBIENT_EFFECTS["neutral"])

    soundtrack = load_audio(soundtrack_file)
    ambient = load_audio(ambient_file)

    # Play soundtrack and ambient effects
    play_audio(soundtrack)
    play_audio(ambient)


# Main storytelling loop with real-time emotion detection
def real_time_emotion_storytelling():
    while True:
        # Get user input for the story
        user_input = input("Enter a story prompt (or type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break

        # Record user voice for emotion detection
        print("Please speak into the microphone to record your emotional state...")
        user_audio = record_audio()

        # Detect user's emotion based on voice
        user_emotion = detect_emotion_from_voice(user_audio)
        print(f"Detected emotion: {user_emotion}")

        # Generate story based on the user's input and emotion
        dynamic_story = generate_narrative(user_emotion, user_input)
        print(f"Generated story: {dynamic_story}")

        # Narrate the generated story with emotion-based modulation
        narrate_story(dynamic_story, user_emotion)

        # Play emotion-based soundtracks and ambient effects
        play_emotion_based_audio(user_emotion)


# Run the real-time emotion-based interactive story experience
if __name__ == "__main__":
    real_time_emotion_storytelling()
"""
Example Inputs and Expected Outputs:
Example 1:
User Input (typed): "A lone traveler enters an abandoned town."
User Voice Emotion: Sad
Detected Emotion: Sad
Generated Story: "The town echoed with silence, as the traveler wandered through the empty streets. A cold wind brushed past, carrying memories of a time long forgotten."
Narration Output:

Voice Modulation: Slow pace, low pitch (sad tone).
Background Music: Soft, melancholic piano music.
Sound Effects: Light rain and soft wind in the background.
Example 2:
User Input (typed): "A hero faces a dragon in a dark cave."
User Voice Emotion: Angry
Detected Emotion: Angry
Generated Story: "The hero clenched their sword, eyes burning with fury. With a deafening roar, the dragon charged, but the hero stood firm, ready to strike."
Narration Output:

Voice Modulation: Fast pace, high pitch (angry tone).
Background Music: Intense battle music.
Sound Effects: Thunderclaps and roaring wind.
Example 3:
User Input (typed): "A group of friends set off on an adventure through a magical forest."
User Voice Emotion: Happy
Detected Emotion: Happy
Generated Story: "The sun gleamed through the trees, as laughter echoed in the air. The friends marveled at the vibrant colors and the enchantment in the air."
Narration Output:

Voice Modulation: Bright tone, medium pace (happy tone).
Background Music: Uplifting, cheerful music.
Sound Effects: Birds chirping and a soft breeze.
Key Features:
Real-Time Emotion Detection: User emotions are detected from their voice input in real-time and drive the adaptive narrative experience.
Adaptive Storyline: The narrative dynamically changes based on the detected emotional tone, offering a personalized and unique experience for each user.
Emotion-Based Audio Modulation: Narrator voice, background music, and ambient effects are adjusted to match the detected emotion, increasing immersion.
Real-Time User Interaction: As the user speaks, their emotional state influences the progression of the story, allowing for an interactive experience where the user’s mood shapes the story.
"""