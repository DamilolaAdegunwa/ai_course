"""
https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
Project Title: Interactive Audio Chatbot with Real-Time Transcription, Emotion Detection, and Response Generation
File Name: interactive_audio_chatbot_with_real_time_transcription_emotion_detection.py

Project Description:
In this advanced project, you'll create an interactive voice-based chatbot. The system will:

Transcribe spoken input in real-time.
Detect emotions in the speaker's input.
Generate dynamic responses based on the content and emotion detected using OpenAI’s GPT model.
Speak the generated response using text-to-speech (TTS) for a full interactive experience.
This chatbot will be able to have a conversation with users by analyzing their tone and emotional context, responding accordingly with personalized replies. The complexity is increased by integrating multiple processes (real-time audio input, emotion analysis, response generation, and audio output).

This project is significantly more advanced due to:

Real-time audio transcription with live feedback.
Emotion detection influencing chatbot responses.
Speech synthesis for a natural, conversational flow.
Continuous interaction (chat loop).
Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey, filepath
import pyaudio
import wave
import pyttsx3  # For text-to-speech
from io import BytesIO
from typing import BinaryIO
# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()
file_path = filepath
# Initialize the Text-to-Speech (TTS) engine
tts_engine = pyttsx3.init()

# Function to transcribe audio and detect emotions
def transcribe_and_detect_emotions(audio_chunk: BinaryIO):
    #audio_file = BytesIO(audio_chunk)

    # Step 1: Transcribe the audio
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_chunk,
        response_format="json"
    )
    transcription = response.text

    # Step 2: Detect emotion from the transcription
    emotion_analysis = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Analyze the emotions in the following text: '{transcription}'. Return the main emotion (e.g., Happy, Sad, Angry, Neutral).",
        max_tokens=30
    )
    emotion = emotion_analysis.choices[0].text.strip()

    return transcription, emotion


# Function to generate a chatbot response based on transcription and emotion
def generate_chatbot_response(transcription, emotion):
    response_generation = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Respond to the following statement, considering the emotion '{emotion}': '{transcription}'.",
        max_tokens=100
    )
    chatbot_response = response_generation.choices[0].text.strip()

    return chatbot_response


# Function to synthesize speech from text
def speak_text(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


# Real-time audio recording function
def record_audio_for_duration(duration=5, chunk_size=1024, channels=1, rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

    print(f"Recording for {duration} seconds...")
    frames = []

    for _ in range(0, int(rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)


# Main function to run the interactive chatbot
def interactive_audio_chatbot():
    print("Welcome to the interactive audio chatbot! Speak your thoughts and get a response!")

    try:
        while True:
            # Step 1: Record user input
            # audio_chunk = record_audio_for_duration(duration=5)
            audio_chunk = open(file_path, "rb")
            # Step 2: Transcribe and detect emotions in the user's speech
            transcription, emotion = transcribe_and_detect_emotions(audio_chunk)

            print(f"\nYou said: {transcription}")
            print(f"Detected emotion: {emotion}")

            # Step 3: Generate chatbot response based on the user's input and emotion
            chatbot_response = generate_chatbot_response(transcription, emotion)

            print(f"Chatbot response: {chatbot_response}")

            # Step 4: Use TTS to speak the chatbot's response
            speak_text(chatbot_response)

            # Exit the chatbot if the user says something like "Goodbye"
            if "goodbye" in transcription.lower() or "bye" in transcription.lower():
                print("Goodbye! Ending the conversation.")
                speak_text("Goodbye! Ending the conversation.")
                break

    except Exception as e:
        print(f"Error: {e}")


# Run the interactive audio chatbot
if __name__ == "__main__":
    interactive_audio_chatbot()
"""
Example Inputs and Expected Outputs:
Example 1:

Input (spoken by the user): "I'm really frustrated that my project isn't going well."
Expected Output:
Transcription: "I'm really frustrated that my project isn't going well."
Detected Emotion: "Frustrated"
Chatbot Response: "I'm sorry to hear you're frustrated. Maybe you could take a short break and come back with fresh ideas?"
TTS Output: Chatbot speaks: "I'm sorry to hear you're frustrated. Maybe you could take a short break and come back with fresh ideas?"
Example 2:

Input (spoken by the user): "I feel so happy today! Everything is going great."
Expected Output:
Transcription: "I feel so happy today! Everything is going great."
Detected Emotion: "Happy"
Chatbot Response: "That's awesome! It’s great to hear that you're having such a good day. Keep enjoying it!"
TTS Output: Chatbot speaks: "That's awesome! It’s great to hear that you're having such a good day. Keep enjoying it!"
Key Concepts and Features:
Real-Time Interaction: The system listens to the user's input, processes it, and responds with a dynamic conversation loop.
Emotion Detection: The chatbot tailors its response based on the emotion detected from the user's voice, making it more empathetic and engaging.
Speech Synthesis (Text-to-Speech): The chatbot not only generates a text-based response but also speaks it out loud, enhancing the interactive experience.
Voice Command to Exit: The system can end the conversation when the user says "Goodbye" or "Bye," adding a more natural flow to the interaction.
This project is more advanced due to the integration of multiple real-time tasks (transcription, emotion detection, response generation, and speech output). It creates a conversational agent capable of understanding and responding to emotional cues, making the interaction more dynamic and personalized.

This system could serve as a foundation for voice-activated personal assistants or even more complex AI-driven conversational systems, making it both challenging and impactful.







"""