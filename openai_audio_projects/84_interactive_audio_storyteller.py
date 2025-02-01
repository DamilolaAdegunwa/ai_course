import os
import openai
import pyttsx3
import random
import json

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load story data
story_data = {
    "start": {
        "text": "You find yourself at a crossroads. Do you want to go to the village or explore the forest?",
        "choices": {
            "village": "village",
            "forest": "forest"
        }
    },
    "village": {
        "text": "You arrive at the village filled with friendly faces. Do you want to visit the market or speak to the village elder?",
        "choices": {
            "market": "market",
            "elder": "elder"
        }
    },
    "forest": {
        "text": "The forest is dark and full of mysterious sounds. Do you want to investigate a noise or keep walking?",
        "choices": {
            "investigate": "investigate",
            "walk": "walk"
        }
    },
    # Additional story segments can be added here
}


# Function to generate story based on user choice
def generate_story_segment(choice):
    prompt = f"Continue the story based on the choice: {choice}."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response['choices'][0]['message']['content']


# Function to narrate story text
def narrate(text):
    engine.say(text)
    engine.runAndWait()


# Function to present choices and get user input
def present_choices(segment):
    print(segment["text"])
    for option in segment["choices"]:
        print(f"- {option.capitalize()}")
    choice = input("What do you choose? ").lower()
    return choice if choice in segment["choices"] else None


# Function to play the interactive story
def play_story():
    current_segment = story_data["start"]

    while True:
        choice = present_choices(current_segment)
        if choice:
            next_segment_key = current_segment["choices"][choice]
            if next_segment_key in story_data:
                current_segment = story_data[next_segment_key]
                # Generate additional story based on choice
                additional_story = generate_story_segment(choice)
                narrate(additional_story)
            else:
                print("This segment has no further choices. Thank you for playing!")
                break
        else:
            print("Invalid choice. Please try again.")


# Main function to start the interactive storytelling
if __name__ == "__main__":
    print("Welcome to the Interactive Audio Storyteller!")
    play_story()
