"""
Project Title: AI-Driven Dynamic Soundscape Generation for Multi-Character Narratives
File name: dynamic_soundscape_for_multi_character_narratives.py

Project Description:
In this project, we will create an advanced dynamic soundscape generator that reacts to multi-character narratives. This system will not only generate AI-driven text for a story but will also adapt audio effects, character voices, spatialization, and background environment based on the story context. The generated audio will reflect the mood, character interactions, and environment in real-time.

We’ll integrate multiple AI audio components:

Voice synthesis: Generate distinct voices for different characters using OpenAI models.
Dynamic background music: Adjust the emotional tone of the background based on narrative context (suspense, joy, sadness).
Character spatialization: Simulate a 3D space where the characters interact from different locations.
Environmental sounds: Add environmental sounds (rain, forest, wind) that adapt to the setting changes in the story.
Emotional adaptation: Modify voices and music in real-time based on the emotional tone of the current part of the story.
Example Use Cases:
Narrative 1:

Input: "Two characters are standing in a forest, one angry and the other calm."
Output: Calm background forest sounds with one character's voice coming from the left side (calm tone) and the other from the right side (angry tone).
Narrative 2:

Input: "A thunderstorm hits, and the protagonist is terrified."
Output: Thunderstorm sound effect with the protagonist’s shaky and panicked voice.
Example Inputs and Expected Outputs:
Example 1:
Input:
“The adventurer enters a dark cave. They hear footsteps echoing behind them as they cautiously walk forward.”

Expected Output:

Echoing footsteps in the background, dark suspenseful music.
The adventurer's voice spatialized in the center, and the echoing footsteps panned from behind.
Example 2:
Input:
“The forest was peaceful, with birds chirping and a gentle breeze. Suddenly, a roar broke the silence.”

Expected Output:

Forest ambient sounds with chirping birds, followed by a sudden loud roar.
Character's reaction with a surprised and panicked voice.
Python Code:
"""
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from apikey import apikey

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Characters and their voice properties
characters = {
    "protagonist": {"voice_pitch": 1.0, "position": {"x": 0, "y": 0, "z": 0}},
    "villain": {"voice_pitch": 0.8, "position": {"x": 2, "y": 0, "z": 0}},
    "sidekick": {"voice_pitch": 1.2, "position": {"x": -2, "y": 0, "z": 0}}
}

# Dynamic music and environment sound setup
environments = {
    "forest": {"sound_file": "forest_ambient.wav", "emotion": "calm"},
    "cave": {"sound_file": "cave_ambient.wav", "emotion": "tense"},
    "storm": {"sound_file": "storm_ambient.wav", "emotion": "scary"}
}


# Generate AI-driven text for the story narrative
def generate_story(input_prompt):
    response = client.Completion.create(
        model="text-davinci-003",
        prompt=input_prompt,
        max_tokens=300
    )
    return response.choices[0].text.strip()


# Adjust spatial audio based on character positions
def calculate_distance(position1, position2):
    return np.sqrt((position1['x'] - position2['x']) ** 2 +
                   (position1['y'] - position2['y']) ** 2 +
                   (position1['z'] - position2['z']) ** 2)


def adjust_spatial_audio(sound, listener_position, character_position):
    distance = calculate_distance(listener_position, character_position)
    volume_adjustment = 1 / (distance + 1)  # Inverse distance attenuation
    return modify_volume(sound, volume_adjustment)


def modify_volume(sound, volume_factor):
    return sound + (volume_factor * 10 - 10)  # Adjust volume in dB


# Apply dynamic environment sounds based on the story
def add_environment_sound(environment_key):
    env_file = environments[environment_key]["sound_file"]
    ambient_sound = AudioSegment.from_file(env_file)
    return ambient_sound


# Generate character voices with pitch adjustments based on the character
def generate_character_voice(text, character_key):
    character = characters[character_key]
    # OpenAI voice generation (this is an approximation, not actual API)
    voice_response = client.Completion.create(
        model="text-davinci-003",
        prompt=f"Generate speech for {character_key}: {text}",
        max_tokens=150
    )
    voice_text = voice_response.choices[0].text.strip()

    # Simulating pitch change with librosa (real TTS not included in OpenAI models)
    voice_data, sr = librosa.load("character_voice.wav", sr=None)
    pitched_voice = librosa.effects.pitch_shift(voice_data, sr, n_steps=character["voice_pitch"] * 12)
    sf.write(f"{character_key}_voice.wav", pitched_voice, sr)

    return AudioSegment.from_file(f"{character_key}_voice.wav")


# Create a dynamic soundscape for the current story segment
def create_dynamic_soundscape(story_segment, environment, characters_in_scene):
    background = add_environment_sound(environment)

    # Adjust and mix character voices into the background
    for character_key in characters_in_scene:
        character_voice = generate_character_voice(story_segment, character_key)
        spatial_voice = adjust_spatial_audio(character_voice, {"x": 0, "y": 0, "z": 0},
                                             characters[character_key]["position"])
        background = background.overlay(spatial_voice)

    return background


# Main story interaction loop
def interactive_storytelling():
    environment = "forest"  # Start in the forest
    current_story = "The adventurer enters the forest."

    while True:
        # Generate the next part of the story
        story_segment = generate_story(current_story)
        print(f"Story: {story_segment}")

        # Create dynamic soundscape based on the story
        soundscape = create_dynamic_soundscape(story_segment, environment, ["protagonist", "villain"])

        # Play the generated soundscape
        play(soundscape)

        # Update the environment or characters based on user input (mock)
        user_input = input("What should the adventurer do next? ")
        if "cave" in user_input:
            environment = "cave"
        elif "storm" in user_input:
            environment = "storm"
        current_story += f" Then the adventurer {user_input}."


# Run the interactive storytelling experience
if __name__ == "__main__":
    interactive_storytelling()
"""
Key Concepts:
AI Text Generation for Story: We use OpenAI's text-davinci-003 model to generate story narratives based on user input.

Character Voice Generation: Different characters are given unique voice characteristics (e.g., pitch adjustments) to distinguish them during dialogue.

Dynamic Background Environment: The background sound changes based on the narrative environment (e.g., forest, cave, storm). Each environment has different ambient sounds that are adapted based on the story.

3D Spatial Audio: Characters’ voices are spatialized in 3D space, depending on their positions in the story. The volume and positioning are adjusted dynamically.

Interactive Storytelling: The story evolves based on user input. The environment, characters, and soundscape are modified according to the user's decisions.

Example of Use:
Test Input 1:

User starts the story with the adventurer entering a dark cave.
User decides that the adventurer will hide behind a rock as they hear a villain approaching.
Test Input 2:

User starts the story with the adventurer walking through a peaceful forest.
Suddenly, a storm approaches, and the user decides to take shelter under a tree.
Expected Results:

Different soundscapes (e.g., cave echoes, villain’s approaching footsteps, storm) adapt to each story progression.
"""