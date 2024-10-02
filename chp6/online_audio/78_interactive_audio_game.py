import os
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


# Function to generate story scenarios using OpenAI
def generate_scenario(choice):
    prompt = f"Generate an audio scenario for a player choosing to {choice}. Include background and effects."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response['choices'][0]['message']['content']


# Function to narrate story scenarios
def narrate_scenario(scenario):
    engine.say(scenario)
    engine.runAndWait()


# Function to add background music
def add_background_music(music_file):
    music = AudioSegment.from_file(music_file)
    music = music - 10  # Reduce music volume
    return music


# Function to add sound effects
def add_sound_effects(effects_files):
    effects = [AudioSegment.from_file(effect_file) for effect_file in effects_files]
    combined_effects = sum(effects)
    return combined_effects


# Function to save final audio
def save_final_audio(final_audio, filename="final_game_audio.mp3"):
    final_audio.export(filename, format="mp3")


# Main function to drive the game
def main():
    print("Welcome to the Interactive Audio Adventure Game!")

    # Initial game state
    game_running = True

    while game_running:
        # Present initial choices to the player
        print("You find yourself at a crossroad. Do you want to enter the dark cave or climb the mountain?")
        choice = input("Enter your choice (cave/mountain): ").lower()

        if choice in ["cave", "mountain"]:
            # Generate scenario based on player choice
            scenario = generate_scenario(choice)
            print("Generating your audio scenario...")
            narrate_scenario(scenario)

            # Add background music and sound effects
            background_music_file = "path/to/background/music.mp3"  # Background music path
            sound_effects_files = ["path/to/sound_effect1.mp3", "path/to/sound_effect2.mp3"]  # Sound effects paths

            music = add_background_music(background_music_file)
            effects = add_sound_effects(sound_effects_files)

            # Combine audio
            combined_audio = AudioSegment.silent(duration=1000)  # Start with 1 second of silence
            combined_audio += music + effects

            # Save final audio
            save_final_audio(combined_audio)
            print("Your scenario audio has been created!")

            # Ask if the player wants to continue
            continue_game = input("Do you want to continue your adventure? (yes/no): ").lower()
            if continue_game != "yes":
                game_running = False
        else:
            print("Invalid choice. Please try again.")

    print("Thank you for playing the Interactive Audio Adventure Game!")


if __name__ == "__main__":
    main()
