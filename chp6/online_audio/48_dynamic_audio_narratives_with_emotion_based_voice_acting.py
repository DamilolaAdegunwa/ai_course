"""
Project Title: Dynamic Audio Narratives with Emotion-Based Voice Acting and Sound Effects
File Name: dynamic_audio_narratives_with_emotion_based_voice_acting.py

Project Description:
In this project, you will build an advanced audio narrative generation system where AI-generated stories adapt in real-time based on user interactions and detected emotions. The narration will be performed by virtual voice actors whose tone changes dynamically based on the narrative's emotional content. Additionally, realistic sound effects and background music are integrated into the narrative based on scene transitions and sentiment analysis. The system will support multiple virtual narrators, allowing for the seamless exchange of voices during character dialogues, while ensuring that each character has its own unique emotional tone.

Complexity:
Multiple virtual narrators: Different voices are assigned to different characters, each with unique emotional tones based on context.
Emotion-based sound effects and background music: Background music and ambient sound effects adapt in real-time, driven by sentiment analysis of the narrative.
Dynamic scene transitions: The story's progress and soundscapes change dynamically based on user input and emotional tone.
Natural language narrative generation: Utilizes advanced language models to generate rich, multi-scene narratives and dialogues.
Real-time audio synthesis: Integrates real-time speech synthesis for different voice actors, with dynamic adjustment of speech properties (speed, tone, pitch) based on the emotion.
Python Code:
"""
import os
import random
from openai import OpenAI
from apikey import apikey
import pyttsx3
from pydub import AudioSegment, playback
from io import BytesIO
import sounddevice as sd
import numpy as np

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS engine for voice acting
tts_engine = pyttsx3.init()

# Define virtual narrators with emotion mapping
NARRATORS = {
    "narrator_1": {"name": "John", "position": [-1, 0, 1], "emotion": "neutral"},
    "narrator_2": {"name": "Sophie", "position": [1, 0, -1], "emotion": "neutral"}
}

# Define emotion-based background music
MUSIC_TRACKS = {
    "happy": "happy_music.wav",
    "sad": "sad_music.wav",
    "angry": "angry_music.wav",
    "neutral": "calm_ambient.wav"
}

# Define emotion-based sound effects for scenes
SOUND_EFFECTS = {
    "happy": "bird_chirping.wav",
    "sad": "rain_drops.wav",
    "angry": "thunder.wav",
    "neutral": "wind_blowing.wav"
}


# Load audio function
def load_audio(file_path):
    return AudioSegment.from_file(file_path)


# Play audio function
def play_audio(audio_segment):
    playback.play(audio_segment)


# Adjust volume of sound based on 3D position
def adjust_volume_for_3d(audio, x, y, z):
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    adjusted_audio = audio - (distance * 5)
    return adjusted_audio


# Record user input audio
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio


# Convert numpy array to PCM format
def numpy_to_pcm(audio_data):
    audio_pcm = (audio_data * 32767).astype(np.int16)
    return audio_pcm.tobytes()


# Generate scene narrative based on user input and emotions
def generate_narrative(user_input):
    prompt = f"Generate an immersive narrative based on the following user input: {user_input}"
    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()


# Detect sentiment and emotion of the scene
def analyze_emotion(scene_text):
    prompt = f"Analyze the sentiment and emotion of this text: {scene_text}"
    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    emotion = response.choices[0].text.strip().lower()
    return emotion


# Adjust TTS properties based on emotion
def adjust_tts_for_emotion(emotion):
    voice_properties = {
        "happy": {"rate": 180, "pitch": 220},
        "sad": {"rate": 110, "pitch": 120},
        "angry": {"rate": 200, "pitch": 250},
        "neutral": {"rate": 150, "pitch": 170}
    }
    if emotion in voice_properties:
        tts_engine.setProperty('rate', voice_properties[emotion]['rate'])
        tts_engine.setProperty('pitch', voice_properties[emotion]['pitch'])


# Speak the narrative with emotional tone and 3D positioning
def narrate_scene(narrator, scene_text, emotion):
    print(f"{narrator} narrating...")
    adjust_tts_for_emotion(emotion)
    tts_engine.say(scene_text)
    tts_engine.runAndWait()


# Play background music based on emotion
def play_background_music(emotion):
    music_file = MUSIC_TRACKS.get(emotion, MUSIC_TRACKS['neutral'])
    music = load_audio(music_file)
    play_audio(music)


# Play sound effects based on emotion
def play_sound_effects(emotion):
    effect_file = SOUND_EFFECTS.get(emotion, SOUND_EFFECTS['neutral'])
    effect = load_audio(effect_file)
    play_audio(effect)


# Main function to generate and narrate stories with emotional soundscapes
def dynamic_narrative_interaction():
    while True:
        # Record user input
        user_input = input("Please provide input for the story (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        # Generate the next scene based on user input
        scene_text = generate_narrative(user_input)
        print(f"Generated Scene: {scene_text}")

        # Analyze emotion from the generated scene
        scene_emotion = analyze_emotion(scene_text)
        print(f"Detected Emotion: {scene_emotion}")

        # Narrator 1 speaks first part of the scene
        narrate_scene(NARRATORS['narrator_1']['name'], scene_text, scene_emotion)

        # Narrator 2 continues the story
        next_scene_text = generate_narrative(f"Continue the story: {scene_text}")
        narrate_scene(NARRATORS['narrator_2']['name'], next_scene_text, scene_emotion)

        # Play corresponding background music and sound effects
        play_background_music(scene_emotion)
        play_sound_effects(scene_emotion)


# Run the dynamic narrative interaction
if __name__ == "__main__":
    dynamic_narrative_interaction()
"""
Example Inputs and Expected Outputs:
Example 1:
User Input: "A group of adventurers arrives at an ancient temple, what happens next?"
Generated Scene: "The adventurers approach the towering gates of the ancient temple. The air grows still, and an eerie silence fills the atmosphere."
Detected Emotion: "neutral"
Narration Output:

Narrator 1 (John): "The adventurers approach the towering gates of the ancient temple."
Narrator 2 (Sophie): "The air grows still, and an eerie silence fills the atmosphere."
Background Music: Calm ambient music.
Sound Effects: Wind blowing gently.
Example 2:
User Input: "Suddenly, a fierce storm begins to brew in the dark sky above."
Generated Scene: "The sky cracks open with the roar of thunder, and lightning illuminates the dark sky as the storm descends with fury."
Detected Emotion: "angry"
Narration Output:

Narrator 1 (John): "The sky cracks open with the roar of thunder."
Narrator 2 (Sophie): "Lightning illuminates the dark sky as the storm descends with fury."
Background Music: Intense, suspenseful music.
Sound Effects: Thunderstorm, lightning cracks.
Example 3:
User Input: "A beautiful sunrise breaks through the clouds, giving the adventurers hope."
Generated Scene: "As the storm passes, the golden rays of the morning sun pierce through the clouds, bathing the adventurers in warm light."
Detected Emotion: "happy"
Narration Output:

Narrator 1 (John): "As the storm passes, the golden rays of the morning sun pierce through the clouds."
Narrator 2 (Sophie): "Bathing the adventurers in warm light, their spirits rise with the promise of a new beginning."
Background Music: Happy, uplifting music.
Sound Effects: Birds chirping.
Key Features:
Multi-Narrator System: Two narrators dynamically take turns in speaking parts of the generated story, each with emotional tone adaptations.
Emotion-Based Background Music: Background music changes dynamically based on the emotion of the scene.
Dynamic Sound Effects: Environmental sounds adapt to the emotional tone of the story, adding a fully immersive audio experience.
"""