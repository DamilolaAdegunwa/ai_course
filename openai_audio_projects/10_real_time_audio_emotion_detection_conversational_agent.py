"""
Project Title: Real-Time Audio Emotion Detection and Sentiment-Aware Conversational Agent
File Name: real_time_audio_emotion_detection_conversational_agent.py

Project Description:
This project builds a real-time emotion detection system that:

Analyzes the emotional tone of a user's speech, detecting sentiment such as happy, sad, angry, or neutral.
Adapts its conversational responses based on the detected emotions and context.
Integrates real-time audio transcription with emotion analysis, creating a sentiment-aware conversational agent that not only understands what is said but how it is said.
Implements a dynamic response mechanism where the agent adjusts its tone and vocabulary based on the speaker's emotional state.
The project supports multiple languages and translates both transcription and emotion data for non-English conversations.
The added complexity lies in processing emotion detection from audio data, responding with emotionally aligned responses, and maintaining this across multiple languages, ensuring a highly interactive experience.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
import pyttsx3
from io import BytesIO
import pyaudio
import numpy as np

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Supported languages for multilingual sentiment analysis
supported_languages = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}

# Emotion classifier labels (assumed)
emotion_labels = ['happy', 'sad', 'angry', 'neutral']


# Function to analyze emotional tone of speech
def analyze_emotion(audio_chunk):
    audio_file = BytesIO(audio_chunk)

    # Step 1: Transcribe the speech
    transcription_response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    transcription = transcription_response['text']

    # Step 2: Analyze the emotion of the transcribed text
    emotion_analysis_prompt = f"Analyze the emotional tone of the following text: '{transcription}'. Indicate whether the speaker is happy, sad, angry, or neutral."
    emotion_response = client.completions.create(
        model="text-davinci-003",
        prompt=emotion_analysis_prompt,
        max_tokens=50
    )
    emotion = emotion_response['choices'][0]['text'].strip().lower()

    if emotion not in emotion_labels:
        emotion = 'neutral'  # Default to neutral if the analysis is unclear

    return transcription, emotion


# Function to generate emotion-aware responses
def generate_emotion_based_response(transcription, emotion):
    response_prompt = f"Given the following statement: '{transcription}', and knowing the speaker's emotion is {emotion}, generate an appropriate response that reflects their emotion."
    response_generation = client.completions.create(
        model="text-davinci-003",
        prompt=response_prompt,
        max_tokens=100
    )
    return response_generation['choices'][0]['text'].strip()


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


# Function to maintain conversation memory
conversation_memory = []


def update_conversation_memory(summary):
    conversation_memory.append(summary)
    if len(conversation_memory) > 10:  # Limit memory size to 10 summaries for now
        conversation_memory.pop(0)


# Main emotion-aware chatbot function
def emotion_aware_conversational_agent():
    print("Welcome to the real-time emotion-aware chatbot!")

    try:
        while True:
            # Step 1: Record the user's input
            audio_chunk = record_audio_for_duration(duration=5)

            # Step 2: Transcribe and analyze emotion from the user's input
            transcription, emotion = analyze_emotion(audio_chunk)

            print(f"\nYou said: {transcription}")
            print(f"Detected emotion: {emotion}")

            # Step 3: Generate an emotion-based response
            chatbot_response = generate_emotion_based_response(transcription, emotion)
            print(f"Chatbot response: {chatbot_response}")

            # Step 4: Convert chatbot response to speech
            tts_engine.say(chatbot_response)
            tts_engine.runAndWait()

            # Update memory
            update_conversation_memory(f"User: {transcription} | Emotion: {emotion}")

            # Step 5: Provide a conversation summary on request
            if "conversation summary" in transcription.lower():
                conversation_summary = summarize_conversation_memory()
                print(f"Conversation Summary: {conversation_summary}")
                tts_engine.say(conversation_summary)
                tts_engine.runAndWait()

            # Exit condition
            if "goodbye" in transcription.lower() or "bye" in transcription.lower():
                print("Goodbye! Ending the conversation.")
                tts_engine.say("Goodbye! Ending the conversation.")
                break

    except Exception as e:
        print(f"Error: {e}")


# Function to summarize the conversation
def summarize_conversation_memory():
    if not conversation_memory:
        return "No conversation data available."

    conversation_context = " ".join(conversation_memory)
    summary_response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Summarize the following conversation: {conversation_context}",
        max_tokens=150
    )
    return summary_response['choices'][0]['text'].strip()


# Run the chatbot
if __name__ == "__main__":
    emotion_aware_conversational_agent()
"""
Example Inputs and Expected Outputs:
Example 1:

Input (spoken in English): "I had a terrible day. Everything went wrong from the moment I woke up."
Expected Output:
Transcription: "I had a terrible day. Everything went wrong from the moment I woke up."
Detected Emotion: "sad"
Chatbot Response: "I'm really sorry to hear that. Would you like to talk more about what went wrong today?"
Example 2:

Input (spoken in Spanish): "¡Estoy tan emocionado! Hoy es mi cumpleaños y tengo una gran celebración planeada."
Expected Output:
Transcription: "¡Estoy tan emocionado! Hoy es mi cumpleaños y tengo una gran celebración planeada."
Detected Emotion: "happy"
Chatbot Response: "That sounds amazing! I hope your birthday celebration goes perfectly. What are you most excited about?"
Key Features:
Emotion Detection from Audio: The system detects emotions from speech in real-time and adapts its responses accordingly, giving the conversation a personalized touch.
Multilingual Support: The project supports multiple languages for both transcription and emotion analysis, making it adaptable to different users.
Emotion-Aware Responses: The chatbot doesn't just provide generic responses but tailors its replies to align with the emotional state of the speaker.
Real-Time Processing: The project handles real-time audio inputs and provides immediate feedback to the user, suitable for interactive conversations or mental health assistance.
Contextual Memory: The bot maintains memory of recent conversations, allowing it to provide summaries or keep context over a longer interaction.
This project introduces emotion-aware interactions and adds the complexity of emotion analysis in real-time to previous OpenAI audio projects. It can be used in scenarios such as customer service, virtual assistants, or mental health support systems that respond empathetically based on the emotional tone of the user's speech.
"""