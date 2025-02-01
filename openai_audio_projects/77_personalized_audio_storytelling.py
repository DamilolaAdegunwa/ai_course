import os
import json
import random
import pyttsx3
import openai
from pydub import AudioSegment
from pydub.playback import play

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize text-to-speech engine
engine = pyttsx3.init()


# Function to generate a story using OpenAI
def generate_story(character, setting, plot, genre):
    prompt = f"Create a {genre} story featuring a character named {character} in {setting}. The plot should involve {plot}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response['choices'][0]['message']['content']


# Function to create audio narration of the story
def narrate_story(story):
    engine.say(story)
    engine.runAndWait()


# Function to add background music to the story
def add_background_music(story_audio, music_file):
    music = AudioSegment.from_file(music_file)
    music = music - 20  # Reduce music volume
    combined = story_audio.overlay(music)
    return combined


# Function to create sound effects for key moments in the story
def add_sound_effects(story_audio, effects_files):
    for effect_file in effects_files:
        effect = AudioSegment.from_file(effect_file)
        story_audio = story_audio.append(effect, crossfade=500)  # Add sound effects with crossfade
    return story_audio


# Function to save the final audio story to a file
def save_audio_story(final_audio, filename="final_story.mp3"):
    final_audio.export(filename, format="mp3")


# Main function to drive the application
def main():
    print("Welcome to the Personalized Audio Storytelling Experience!")

    # Gather user input
    character = input("Enter a character name: ")
    setting = input("Enter a setting: ")
    plot = input("Enter a plot element: ")
    genre = input("Choose a genre (fantasy, adventure, mystery): ")

    # Generate story
    story = generate_story(character, setting, plot, genre)
    print("Generating your story...")

    # Narrate story
    narrate_story(story)

    # Create audio from the narration
    audio_segment = AudioSegment.from_mono_audiosegments(engine)

    # Add background music
    background_music_file = "path/to/background/music.mp3"  # Path to your background music file
    audio_with_music = add_background_music(audio_segment, background_music_file)

    # Add sound effects for key moments
    sound_effects_files = ["path/to/effect1.mp3", "path/to/effect2.mp3"]  # Add paths to sound effects
    final_audio = add_sound_effects(audio_with_music, sound_effects_files)

    # Save final audio story
    save_audio_story(final_audio)
    print("Your personalized audio story has been created!")


if __name__ == "__main__":
    main()
