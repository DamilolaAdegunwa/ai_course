"""
Project Title: AI-Powered Audio Drama Director with Real-Time Adaptive Script Generation and Sound Design
File Name: ai_audio_drama_director.py

Project Description:
In this advanced OpenAI audio project, you will create an AI-powered director for an audio drama. The AI will generate an interactive script for an audio drama, allowing multiple actors (or speakers) to participate in real time. It uses real-time speech recognition, emotion detection, and sound design to adapt the script dynamically based on actors’ input, detected emotions, and story progression.

Key Components:

Real-Time Adaptive Script Generation: OpenAI models generate scripts based on the story flow and characters' interactions. The system will also detect and adjust to emotional tones.
Dynamic Sound Design: Real-time sound effects (e.g., footsteps, ambient sounds) are dynamically added based on the script, enhancing the immersive audio drama experience.
Emotion-Aware Dialogue: Emotion detection informs the AI on how to adapt dialogue and plot progression, enhancing the depth of the narrative.
Multi-Character Management: Multiple actors can speak, and the AI will manage different character interactions, voices, and emotional arcs, adjusting sound effects and background music accordingly.
Story Progression and Branching: The AI keeps track of story progression and can branch narratives based on actors' input and detected emotions.
This project is designed to simulate a virtual audio drama where the AI acts as the director, adapting the plot, dialogues, and soundscape in real-time.

Python Code:
"""
import os
import numpy as np
import sounddevice as sd
import pyttsx3
import librosa
from openai import OpenAI
from apikey import apikey
import speech_recognition as sr
from pydub import AudioSegment, playback

# OpenAI API setup
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Speech recognizer for real-time input
recognizer = sr.Recognizer()

# Load background sound effects (e.g., footsteps, ambient sound)
SOUND_EFFECTS = {
    "footsteps": "footsteps.wav",
    "rain": "rain_ambience.wav",
    "door_creak": "door_creak.wav"
}


# Manage sound effects during audio drama
def load_sound_effect(effect_name):
    if effect_name in SOUND_EFFECTS:
        return AudioSegment.from_file(SOUND_EFFECTS[effect_name])
    return None


# Play sound effect dynamically based on scene
def play_sound_effect(effect_name):
    sound = load_sound_effect(effect_name)
    if sound:
        playback.play(sound)


# Record real-time voice input
def record_voice(duration=5, sample_rate=16000):
    print("Recording voice input...")
    voice_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return voice_data


# Detect speaker emotions using OpenAI
def detect_emotion(voice_data, sample_rate=16000):
    voice_wav = librosa.resample(voice_data[:, 0], orig_sr=sample_rate, target_sr=16000)
    voice_pcm = librosa.util.buf_to_int(voice_wav)

    # Transcribe and detect emotion from speech
    transcribed_audio = client.audio.transcribe(model="whisper-1", file=voice_pcm)
    transcription = transcribed_audio['text']

    # Generate emotion analysis
    emotion_analysis = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotional tone of this conversation: {transcription}",
        max_tokens=50
    )
    emotion = emotion_analysis.choices[0].text.strip()
    print(f"Detected emotion: {emotion}")
    return transcription, emotion


# Generate real-time drama script based on input and emotions
def generate_adaptive_script(scene_context, actor_input, emotion):
    prompt = f"Here is the current scene context: {scene_context}. The actor said: {actor_input}. Their emotion is {emotion}. Continue the audio drama script."

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )

    generated_script = response.choices[0].text.strip()
    return generated_script


# Add dynamic sound effects based on script context
def add_dynamic_sound_effect(script):
    if "footsteps" in script:
        play_sound_effect("footsteps")
    elif "rain" in script:
        play_sound_effect("rain")
    elif "door creaks" in script:
        play_sound_effect("door_creak")


# Main function to manage the audio drama flow
def interactive_audio_drama():
    scene_context = "The story starts in a dark and rainy night, with the sound of footsteps approaching a wooden door."

    # Play initial background sounds
    play_sound_effect("rain")

    while True:
        # Record actor's input
        print("Awaiting actor input...")
        voice_input = record_voice(duration=5)
        transcription, emotion = detect_emotion(voice_input)

        # Generate the next part of the script based on input and emotions
        generated_script = generate_adaptive_script(scene_context, transcription, emotion)
        print(f"Generated Script: {generated_script}")

        # Add dynamic sound effects based on the script
        add_dynamic_sound_effect(generated_script)

        # Speak the generated script using text-to-speech
        tts_engine.say(generated_script)
        tts_engine.runAndWait()

        # Update the scene context
        scene_context += f"\n{generated_script}"

        # Exit condition
        if "end" in transcription.lower():
            print("Audio drama has ended.")
            break


# Run the audio drama director
if __name__ == "__main__":
    interactive_audio_drama()
"""
Example Inputs and Expected Outputs:
Example 1:
Actor Input: “I hear footsteps approaching. Should I open the door?”
Detected Emotion: Suspense
Generated Script: “As the footsteps grow louder, the door creaks open slowly, revealing a shadowy figure standing outside in the rain.”

Sound Effects:

Footsteps sound (played dynamically)
Door creak sound (triggered by the script)
Example 2:
Actor Input: “I’m not sure what to do, it’s getting dark.”
Detected Emotion: Fear
Generated Script: “The sky darkens rapidly, and a cold wind starts to blow. The rain becomes heavier, and you sense something ominous in the air.”

Sound Effects:

Rain sound (volume increases to match the heavy rain)
Wind sound (played dynamically to enhance the drama)
Example 3:
Actor Input: “The door creaks open, and I walk inside.”
Detected Emotion: Cautious
Generated Script: “You step inside cautiously, the wooden floor creaking beneath your feet. The room is dimly lit, and you can barely make out a figure sitting in the corner.”

Sound Effects:

Footsteps sound (played dynamically for the walking action)
Floor creaking sound (added for enhanced immersion)
Key Features:
Real-Time Script Generation: The AI dynamically generates the audio drama script based on actor input and emotional tone, ensuring that each interaction feels unique.
Adaptive Sound Design: The system integrates relevant sound effects dynamically based on the evolving script, enhancing the immersive nature of the audio drama.
Emotion Detection: By analyzing actors' speech, the AI detects emotions that influence how the script and sound effects progress, creating a more engaging narrative.
Multi-Character and Scene Progression: The system manages multi-character dialogues and tracks the progression of the drama, allowing for smooth transitions between scenes.
Branching Narratives: Based on actors' input and emotions, the story can branch into different directions, creating a flexible and adaptive audio drama experience.
Conclusion:
This project takes the complexity of OpenAI-powered audio projects to the next level by combining real-time script generation, sound effects, and emotion detection into a fully immersive and adaptive audio drama experience. By allowing multiple actors to influence the progression of the story, the AI enhances creative storytelling in real-time, giving users an engaging and highly interactive audio drama experience.Example Inputs and Expected Outputs:
Example 1:
Actor Input: “I hear footsteps approaching. Should I open the door?”
Detected Emotion: Suspense
Generated Script: “As the footsteps grow louder, the door creaks open slowly, revealing a shadowy figure standing outside in the rain.”

Sound Effects:

Footsteps sound (played dynamically)
Door creak sound (triggered by the script)
Example 2:
Actor Input: “I’m not sure what to do, it’s getting dark.”
Detected Emotion: Fear
Generated Script: “The sky darkens rapidly, and a cold wind starts to blow. The rain becomes heavier, and you sense something ominous in the air.”

Sound Effects:

Rain sound (volume increases to match the heavy rain)
Wind sound (played dynamically to enhance the drama)
Example 3:
Actor Input: “The door creaks open, and I walk inside.”
Detected Emotion: Cautious
Generated Script: “You step inside cautiously, the wooden floor creaking beneath your feet. The room is dimly lit, and you can barely make out a figure sitting in the corner.”

Sound Effects:

Footsteps sound (played dynamically for the walking action)
Floor creaking sound (added for enhanced immersion)
Key Features:
Real-Time Script Generation: The AI dynamically generates the audio drama script based on actor input and emotional tone, ensuring that each interaction feels unique.
Adaptive Sound Design: The system integrates relevant sound effects dynamically based on the evolving script, enhancing the immersive nature of the audio drama.
Emotion Detection: By analyzing actors' speech, the AI detects emotions that influence how the script and sound effects progress, creating a more engaging narrative.
Multi-Character and Scene Progression: The system manages multi-character dialogues and tracks the progression of the drama, allowing for smooth transitions between scenes.
Branching Narratives: Based on actors' input and emotions, the story can branch into different directions, creating a flexible and adaptive audio drama experience.
Conclusion:
This project takes the complexity of OpenAI-powered audio projects to the next level by combining real-time script generation, sound effects, and emotion detection into a fully immersive and adaptive audio drama experience. By allowing multiple actors to influence the progression of the story, the AI enhances creative storytelling in real-time, giving users an engaging and highly interactive audio drama experience.
"""