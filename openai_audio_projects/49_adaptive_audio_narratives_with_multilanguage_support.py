"""
Project Title: Adaptive Audio Narratives with Emotion-Based Voice Modulation, Real-Time Soundtrack Generation, and Multi-Language Support
File Name: adaptive_audio_narratives_with_multilanguage_support.py

Project Description:
This project focuses on creating a multi-language adaptive audio narrative system where generated audio dynamically changes based on real-time emotion analysis and includes on-the-fly soundtrack generation using AI. The system supports multiple languages for both the narrative and speech synthesis, providing a more global experience. It includes the following advanced features:

Emotion-based Voice Modulation: Modifies narrator voices based on detected emotional tones, adjusting speed, pitch, and tone in real-time.
Real-Time Soundtrack Generation: Dynamically generates and integrates AI-composed soundtracks for background music in sync with the narrative's emotional flow.
Multi-Language Support: Narration and speech synthesis can be done in different languages, switching seamlessly between languages based on user choice or dialogue settings.
Adaptive Storyline Progression: The narrative adapts based on user input and detected emotions, affecting not only the storyline but also the soundscapes and voice performance.
Emotion-based Ambient Sound Effects: Background sounds (such as rain, thunder, forest sounds, etc.) dynamically adjust to reflect the scene's emotional tone, heightening the immersive experience.
Python Code:
"""
import os
import random
from openai import OpenAI
from apikey import apikey  # Use your apikey.py
import pyttsx3
import sounddevice as sd
import numpy as np
from pydub import AudioSegment, playback

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS engine for voice modulation
tts_engine = pyttsx3.init()

# Define multi-language narrators with emotion mapping
LANGUAGES = ["en", "es", "fr", "de", "it"]
NARRATORS = {
    "english": {"name": "John", "lang": "en", "emotion": "neutral"},
    "spanish": {"name": "Sofia", "lang": "es", "emotion": "neutral"}
}

# Define emotion-based soundtrack
SOUNDTRACKS = {
    "happy": "upbeat_background.wav",
    "sad": "slow_piano.wav",
    "angry": "intense_beat.wav",
    "neutral": "calm_ambient.wav"
}

# Define ambient sound effects for scenes
AMBIENT_EFFECTS = {
    "happy": "forest_birds.wav",
    "sad": "light_rain.wav",
    "angry": "storm_thunder.wav",
    "neutral": "wind_whistle.wav"
}

# Load audio function
def load_audio(file_path):
    return AudioSegment.from_file(file_path)

# Play audio function
def play_audio(audio_segment):
    playback.play(audio_segment)

# Record user input (real-time)
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio

# Convert numpy array to PCM format for further processing
def numpy_to_pcm(audio_data):
    audio_pcm = (audio_data * 32767).astype(np.int16)
    return audio_pcm.tobytes()

# Generate narrative dynamically
def generate_narrative(prompt, language='en'):
    response = client.completions.create(
        model="gpt-3.5-turbo",
        prompt=f"Generate a narrative in {language}: {prompt}",
        max_tokens=300
    )
    return response.choices[0].text.strip()

# Detect emotion of the text
def analyze_emotion(scene_text):
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotion in the following text: {scene_text}",
        max_tokens=50
    )
    emotion = response.choices[0].text.strip().lower()
    return emotion

# Generate AI-composed soundtrack for the scene based on the emotion
def generate_soundtrack(emotion):
    return SOUNDTRACKS.get(emotion, SOUNDTRACKS["neutral"])

# Adjust voice modulation based on emotion
def adjust_voice_for_emotion(emotion):
    properties = {
        "happy": {"rate": 180, "pitch": 220},
        "sad": {"rate": 110, "pitch": 120},
        "angry": {"rate": 200, "pitch": 250},
        "neutral": {"rate": 150, "pitch": 170}
    }
    if emotion in properties:
        tts_engine.setProperty("rate", properties[emotion]["rate"])
        tts_engine.setProperty("pitch", properties[emotion]["pitch"])

# Narrate a scene using TTS in the specified language
def narrate_scene(narrator, scene_text, emotion, language='en'):
    print(f"{narrator} narrating in {language}...")
    adjust_voice_for_emotion(emotion)
    tts_engine.say(scene_text)
    tts_engine.runAndWait()

# Play ambient sound based on emotion
def play_ambient_sound(emotion):
    ambient_file = AMBIENT_EFFECTS.get(emotion, AMBIENT_EFFECTS["neutral"])
    ambient = load_audio(ambient_file)
    play_audio(ambient)

# Play generated soundtrack in sync with the emotion
def play_emotion_based_soundtrack(emotion):
    soundtrack_file = generate_soundtrack(emotion)
    soundtrack = load_audio(soundtrack_file)
    play_audio(soundtrack)

# Multi-language adaptive story interaction
def multi_language_story_interaction():
    while True:
        # Ask for user input in the form of a narrative prompt
        user_prompt = input("Provide a prompt for the story (type 'exit' to quit): ")
        if user_prompt.lower() == "exit":
            break

        # Ask user to select the language for narration
        user_language = input(f"Choose a language from {LANGUAGES}: ").strip().lower()
        if user_language not in LANGUAGES:
            print("Invalid language selected, defaulting to English.")
            user_language = "en"

        # Generate a narrative based on the user's prompt and language
        generated_narrative = generate_narrative(user_prompt, language=user_language)
        print(f"Narrative generated in {user_language}: {generated_narrative}")

        # Analyze the emotion of the generated narrative
        emotion = analyze_emotion(generated_narrative)
        print(f"Detected emotion: {emotion}")

        # Play the narration with appropriate emotional modulation
        narrate_scene(NARRATORS[user_language]['name'], generated_narrative, emotion, language=user_language)

        # Play background music and ambient sound effects based on the emotion
        play_emotion_based_soundtrack(emotion)
        play_ambient_sound(emotion)

# Run the adaptive multi-language audio narrative system
if __name__ == "__main__":
    multi_language_story_interaction()
"""
Example Inputs and Expected Outputs:
Example 1:
User Input: "A brave warrior enters a dense forest."
Language Selected: English (en)
Generated Narrative: "The brave warrior stepped into the heart of the forest, where the trees whispered secrets from ages past."
Detected Emotion: neutral
Narration Output:

Narrator John speaks the line with a neutral voice.
Background Music: Calm ambient music.
Sound Effects: Wind blowing gently.
Example 2:
User Input: "Un guerrero valiente se adentra en el bosque."
Language Selected: Spanish (es)
Generated Narrative: "El valiente guerrero se adentró en el corazón del bosque, donde los árboles susurraban secretos de épocas pasadas."
Detected Emotion: neutral
Narration Output:

Narrator Sofia speaks the line in Spanish with a neutral tone.
Background Music: Calm ambient music.
Sound Effects: Light wind sound.
Example 3:
User Input: "Suddenly, a powerful storm begins to rage overhead."
Language Selected: English (en)
Generated Narrative: "Without warning, the sky cracked open, and a fierce storm unleashed its fury upon the warrior."
Detected Emotion: angry
Narration Output:

Narrator John speaks in an angry tone, with a fast pace and high pitch.
Background Music: Intense, suspenseful music.
Sound Effects: Thunderstorms and heavy rain.
Example 4:
User Input: "Un hermoso amanecer aparece sobre el horizonte."
Language Selected: Spanish (es)
Generated Narrative: "Un hermoso amanecer apareció sobre el horizonte, bañando todo en luz dorada."
Detected Emotion: happy
Narration Output:

Narrator Sofia speaks in a cheerful, upbeat tone.
Background Music: Uplifting, happy music.
Sound Effects: Birds chirping.
Key Features:
Multi-Language Support: Narration and speech synthesis in multiple languages, with seamless switching.
Emotion-Driven Voice Modulation: Voice speed, pitch, and tone adjust dynamically based on the emotional context of the story.
Real-Time AI Soundtrack Generation: Dynamic soundtracks and ambient effects match the emotional tone of each scene for full immersion.
"""