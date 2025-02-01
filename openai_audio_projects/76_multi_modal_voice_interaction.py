import os
import json
import speech_recognition as sr
import pyttsx3
import openai
from datetime import datetime
from random import choice

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()


# Load user preferences from a JSON file
def load_user_preferences():
    try:
        with open("user_preferences.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# Save user preferences to a JSON file
def save_user_preferences(preferences):
    with open("user_preferences.json", "w") as f:
        json.dump(preferences, f)


# Function to recognize speech input
def listen_for_commands():
    with sr.Microphone() as source:
        print("Listening for commands...")
        audio_data = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio_data)
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I could not understand your command.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None


# Function to respond with audio
def respond_with_audio(response):
    engine.say(response)
    engine.runAndWait()


# Generate personalized audio recommendations
def generate_audio_recommendation(preferences):
    mood = preferences.get("mood", "neutral")
    time_of_day = datetime.now().hour
    greeting = "Good morning!" if time_of_day < 12 else "Good evening!"

    if mood == "happy":
        recommendation = "How about some upbeat music to start your day?"
    elif mood == "relaxed":
        recommendation = "Let's listen to some calming sounds."
    else:
        recommendation = "Here are some podcasts you might enjoy."

    return f"{greeting} {recommendation}"


# Generate summary of a podcast episode
def generate_podcast_summary(podcast_name):
    prompt = f"Provide a summary for the latest episode of the podcast '{podcast_name}'."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response['choices'][0]['message']['content']


# Main interaction loop
def main():
    preferences = load_user_preferences()

    while True:
        command = listen_for_commands()

        if command:
            command = command.lower()
            if "music" in command or "playlist" in command:
                preferences["mood"] = "happy"  # Example: set mood to happy for music
                recommendation = generate_audio_recommendation(preferences)
                respond_with_audio(recommendation)

            elif "podcast" in command:
                podcast_name = "Your Favorite Podcast"  # Replace with actual podcast
                summary = generate_podcast_summary(podcast_name)
                respond_with_audio(summary)

            elif "goodbye" in command:
                respond_with_audio("Goodbye! Have a great day!")
                break

    save_user_preferences(preferences)


if __name__ == "__main__":
    main()
