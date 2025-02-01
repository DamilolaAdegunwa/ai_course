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

# Define the game rooms and their puzzles
rooms = {
    "entrance": {
        "description": "You are at the entrance of the escape room. There is a door to the east.",
        "puzzle": "What has keys but can't open locks?",
        "answer": "piano",
        "hint": "It's a musical instrument.",
        "next_room": "library"
    },
    "library": {
        "description": "You enter a dusty library filled with books. There's a mysterious book on the table.",
        "puzzle": "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?",
        "answer": "echo",
        "hint": "It's a phenomenon related to sound.",
        "next_room": "treasure_room"
    },
    "treasure_room": {
        "description": "Congratulations! You've found the treasure room filled with gold and jewels!",
        "puzzle": None,
        "answer": None,
        "hint": None,
        "next_room": None
    }
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
    current_room = "entrance"
    speak(rooms[current_room]["description"])  # Describe the starting location

    while current_room:
        if rooms[current_room]["puzzle"]:
            speak("You have a puzzle to solve.")
            speak(rooms[current_room]["puzzle"])

        command = get_player_input()
        if command:
            if "look at" in command:
                speak(rooms[current_room]["description"])
                dialogue = generate_dialogue("looking at the painting or object")
                speak(dialogue)

            elif "hint" in command:
                speak(rooms[current_room]["hint"])

            elif "solve puzzle" in command:
                speak("What is your answer?")
                answer = get_player_input()
                if answer == rooms[current_room]["answer"]:
                    speak("Correct! You may proceed to the next room.")
                    current_room = rooms[current_room]["next_room"]
                    if current_room:
                        speak(rooms[current_room]["description"])
                    else:
                        speak("You've escaped the room! Congratulations!")
                else:
                    speak("That's not correct. Try again!")

            elif command == "quit":
                speak("Thank you for playing!")
                break

            else:
                speak("I didn't quite get that. You can say 'look at', 'hint', or 'solve puzzle'.")


# Main function to start the game
if __name__ == "__main__":
    print("Welcome to the Audio-Enhanced Virtual Escape Room!")
    play_game()
"""
https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""