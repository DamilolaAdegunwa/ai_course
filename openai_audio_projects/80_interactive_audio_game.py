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

# Define a simple map of the game
game_map = {
    "start": "You are at the starting point. There are paths to the north and east.",
    "north": "You walk north and find a mysterious cave.",
    "east": "You head east and come across a river.",
    "cave": "Inside the cave, you hear strange noises. There's a puzzle to solve!",
}

# Function to generate interactive dialogue based on player actions
def generate_dialogue(action):
    prompt = f"A character reacts to the player's action of {action}. Provide a dialogue."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response['choices'][0]['message']['content']

# Function to speak out text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to capture player voice input
def get_player_input():
    with sr.Microphone() as source:
        print("Listening for your command...")
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

# Function to play the game
def play_game():
    current_location = "start"
    speak(game_map[current_location])  # Describe the starting location

    while True:
        command = get_player_input()
        if command:
            if "go north" in command:
                current_location = "north"
                speak(game_map[current_location])
                dialogue = generate_dialogue("going north")
                speak(dialogue)

            elif "go east" in command:
                current_location = "east"
                speak(game_map[current_location])
                dialogue = generate_dialogue("going east")
                speak(dialogue)

            elif "solve puzzle" in command and current_location == "north":
                # Example puzzle interaction
                speak("What is the answer to the ultimate question of life, the universe, and everything?")
                answer = get_player_input()
                if answer == "42":
                    speak("Correct! You may proceed deeper into the cave.")
                    current_location = "cave"
                    speak(game_map[current_location])
                else:
                    speak("That's not correct. Try again!")

            elif command == "quit":
                speak("Thank you for playing!")
                break

            else:
                speak("I didn't quite get that. You can say 'go north', 'go east', or 'solve puzzle'.")

# Main function to start the game
if __name__ == "__main__":
    print("Welcome to the Interactive Audio Game!")
    play_game()
