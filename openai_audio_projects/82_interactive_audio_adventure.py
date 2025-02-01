import os
import openai
import speech_recognition as sr
import pyttsx3
import random
import time

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Define the story structure
story_segments = {
    "start": {
        "description": "You find yourself at the edge of a mysterious forest. Do you wish to enter the forest or walk along the path?",
        "choices": {
            "enter": "forest",
            "walk": "path"
        }
    },
    "forest": {
        "description": "The forest is dark and filled with strange noises. You see a cave ahead. Do you want to enter the cave or go deeper into the forest?",
        "choices": {
            "enter": "cave",
            "deeper": "deep_forest"
        }
    },
    "cave": {
        "description": "Inside the cave, you find a sleeping dragon. What do you want to do?",
        "choices": {
            "sneak": "sneak_away",
            "attack": "dragon"
        }
    },
    "deep_forest": {
        "description": "You encounter a wizard who offers you a choice. Do you want to accept his offer or refuse?",
        "choices": {
            "accept": "wizard_offer",
            "refuse": "refusal"
        }
    },
    "sneak_away": {
        "description": "You quietly sneak away, but the dragon wakes up and chases you!",
        "choices": {
            "run": "escape",
            "fight": "dragon"
        }
    },
    "wizard_offer": {
        "description": "The wizard grants you a magical power! What will you do with it?",
        "choices": {
            "explore": "explore",
            "attack": "wizard"
        }
    },
    "dragon": {
        "description": "You bravely attack the dragon and manage to slay it! You find a treasure chest. What do you do?",
        "choices": {
            "open": "treasure",
            "leave": "leave_treasure"
        }
    },
    "treasure": {
        "description": "You open the treasure chest and find gold! Congratulations, you have completed the adventure!",
        "choices": {}
    },
    "leave_treasure": {
        "description": "You decide to leave the treasure behind and exit the cave. Your adventure ends here.",
        "choices": {}
    },
    "escape": {
        "description": "You manage to escape from the dragon but lose your way in the forest. Your adventure ends here.",
        "choices": {}
    },
    "refusal": {
        "description": "The wizard gets angry and disappears in a puff of smoke. Your adventure ends here.",
        "choices": {}
    }
}


# Function to generate interactive dialogue based on user choices
def generate_dialogue(action):
    prompt = f"A character reacts to the player's choice of {action}. Provide a dialogue."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']


# Function to speak out text
def speak(text):
    engine.say(text)
    engine.runAndWait()


# Function to capture player voice input
def get_player_input():
    with sr.Microphone() as source:
        print("Listening for your choice...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        print(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


# Function to play the interactive adventure
def play_adventure():
    current_segment = "start"

    while current_segment:
        # Narrate the current segment
        speak(story_segments[current_segment]["description"])

        # Get player's choice
        command = get_player_input()
        if command:
            # Check if the command matches any choices
            for choice, next_segment in story_segments[current_segment]["choices"].items():
                if choice in command:
                    current_segment = next_segment
                    dialogue = generate_dialogue(f"{choice} choice in {current_segment}")
                    speak(dialogue)
                    break
            else:
                speak("I didn't quite get that. Please choose again.")
        else:
            speak("I didn't hear you. Please try again.")


# Main function to start the adventure
if __name__ == "__main__":
    print("Welcome to the Personalized Interactive Audio Adventure!")
    play_adventure()
