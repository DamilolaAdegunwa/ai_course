import os
import openai
import sounddevice as sd
import simpleaudio as sa
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
openai.api_key = os.getenv("OPENAI_API_KEY")


# Function to generate story using OpenAI
def generate_story(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response['choices'][0]['message']['content']


# Function to generate audio for storytelling
def generate_audio(text, filename="story_audio.wav"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    # Play the audio using simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()  # Wait until audio is finished playing


# Function to record user speech
def record_audio(duration=5):
    print("Recording...")
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return recording


# Function to recognize speech and return the text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition service."


# Main function to run the interactive storytelling assistant
def run_storytelling_assistant():
    print("Welcome to the Interactive Storytelling Assistant!")

    # Get user preferences
    print("What genre of story would you like? (e.g., fantasy, adventure, mystery)")
    genre = recognize_speech()
    print("You chose:", genre)

    print("What setting would you like? (e.g., forest, castle, space)")
    setting = recognize_speech()
    print("You chose:", setting)

    print("Who is the main character? (e.g., a brave knight, a clever detective)")
    character = recognize_speech()
    print("You chose:", character)

    # Generate the story prompt
    story_prompt = f"Create a {genre} story set in a {setting} featuring a character who is {character}."
    story = generate_story(story_prompt)
    print("Here's your story:", story)

    # Generate audio for the story
    generate_audio(story)

    # User interaction for choices
    print("Would you like to make a choice in the story? (yes/no)")
    response = recognize_speech()

    if "yes" in response:
        print("What choice would you like to make?")
        user_choice = recognize_speech()
        choice_prompt = f"{story} The user chooses: {user_choice}. What happens next?"
        next_part = generate_story(choice_prompt)
        print("Next part of your story:", next_part)

        # Generate audio for the next part
        generate_audio(next_part)

    print("Thank you for using the Interactive Storytelling Assistant!")


# Main entry point
if __name__ == "__main__":
    run_storytelling_assistant()
