"""
Project Title: Emotionally-Responsive Multilingual Audio Chatbot with Real-Time Emotion Detection and Sentiment Analysis
File Name: emotionally_responsive_multilingual_audio_chatbot.py

Project Description:
In this advanced project, you’ll build an emotionally-responsive conversational chatbot that:

Transcribes and translates multilingual audio input in real-time.
Analyzes the emotional tone and sentiment (e.g., happy, sad, neutral, frustrated) of the user's speech using OpenAI models.
Adapts its responses dynamically based on the detected emotion to create a more empathetic and personalized interaction.
Integrates text-to-speech (TTS) for real-time spoken responses.
Provides a summary of the emotional state throughout the conversation on request.
This project adds emotion and sentiment analysis, which increases complexity by adding a more human-like layer of interaction, making the chatbot capable of emotionally-aware conversations.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
from io import BytesIO
import pyaudio
import pyttsx3

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Conversation history and emotion tracking
conversation_history = []
emotion_summary = []

# Supported languages
supported_languages = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}

# Function to transcribe, translate, and analyze emotion from audio input
def transcribe_translate_analyze_emotion(audio_chunk, target_language="en"):
    audio_file = BytesIO(audio_chunk)

    # Step 1: Transcribe the audio input
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    transcription = response['text']
    detected_language = response.get('language', 'en')  # Default to English

    # Step 2: Translate to English or desired language
    translation = client.translations.create(
        model="whisper-1",
        source_language=detected_language,
        target_language=target_language,
        text=transcription
    )
    translated_text = translation['translated_text']

    # Step 3: Analyze emotion using OpenAI's sentiment/emotion model
    emotion_analysis = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotional tone of the following text: '{translated_text}'. Classify it as happy, sad, neutral, frustrated, or angry.",
        max_tokens=30
    )
    detected_emotion = emotion_analysis['choices'][0]['text'].strip().lower()

    return transcription, translated_text, detected_emotion, detected_language

# Function to generate an emotionally aware response
def generate_emotionally_aware_response(user_input, emotion):
    prompt = (
        f"Respond to the following input: '{user_input}', considering the emotion '{emotion}'. "
        f"Adjust the tone accordingly, offering support if the user is sad or frustrated, encouragement if they are happy, etc."
    )
    response_generation = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response_generation['choices'][0]['text'].strip()

# Real-time audio recording
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

# Function to summarize the emotions throughout the conversation
def summarize_emotions(emotion_summary):
    prompt = f"Summarize the following emotional states from a conversation: {', '.join(emotion_summary)}."
    summary_response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    return summary_response['choices'][0]['text'].strip()

# Main chatbot function
def emotionally_responsive_chatbot():
    print("Welcome to the emotionally responsive chatbot!")

    try:
        while True:
            # Step 1: Record user input
            audio_chunk = record_audio_for_duration(duration=5)

            # Step 2: Transcribe, translate, and analyze the user's input
            transcription, translated_text, detected_emotion, detected_language = transcribe_translate_analyze_emotion(audio_chunk)

            print(f"\nYou said (in {supported_languages.get(detected_language, 'unknown language')}): {transcription}")
            print(f"Translated to English: {translated_text}")
            print(f"Detected Emotion: {detected_emotion}")

            # Save emotion summary for the conversation
            emotion_summary.append(detected_emotion)

            # Step 3: Generate an emotionally aware response
            chatbot_response = generate_emotionally_aware_response(translated_text, detected_emotion)
            print(f"Chatbot response: {chatbot_response}")

            # Save conversation history
            conversation_history.append({
                "user_input": translated_text,
                "response": chatbot_response,
                "emotion": detected_emotion
            })

            # Step 4: Convert chatbot response to speech
            tts_engine.say(chatbot_response)
            tts_engine.runAndWait()

            # Step 5: Check if the user requests an emotion summary
            if "summary" in transcription.lower():
                summary = summarize_emotions(emotion_summary)
                print(f"Emotion summary: {summary}")
                tts_engine.say(summary)
                tts_engine.runAndWait()

            # Exit condition
            if "goodbye" in transcription.lower() or "bye" in transcription.lower():
                print("Goodbye! Ending the conversation.")
                tts_engine.say("Goodbye! Ending the conversation.")
                break

    except Exception as e:
        print(f"Error: {e}")

# Run the chatbot
if __name__ == "__main__":
    emotionally_responsive_chatbot()
"""
Example Inputs and Expected Outputs:
Example 1:

Input (spoken in French, with a sad tone): "Je me sens très seul aujourd'hui."
Expected Output:
Transcription: "Je me sens très seul aujourd'hui."
Translated to English: "I feel very lonely today."
Emotion: "sad"
Chatbot Response: "I'm really sorry to hear that you're feeling lonely. Do you want to talk about it?"
Emotion Summary (if requested): "The user has expressed sadness throughout the conversation, mentioning feelings of loneliness."
Example 2:

Input (spoken in Spanish, with a happy tone): "¡Hoy es un gran día para mí!"
Expected Output:
Transcription: "¡Hoy es un gran día para mí!"
Translated to English: "Today is a great day for me!"
Emotion: "happy"
Chatbot Response: "That's amazing to hear! What has made your day so wonderful?"
Emotion Summary (if requested): "The user has been feeling happy throughout the conversation, expressing excitement and joy."
Key Features:
Emotionally-Aware Responses: The chatbot dynamically adjusts its tone and response style based on the detected emotion (e.g., sympathetic for sadness, enthusiastic for happiness).
Real-Time Emotion Detection: The chatbot analyzes the user's emotional tone from their speech to create an empathetic interaction.
Multilingual Support: Users can interact in various languages, and the bot will transcribe and translate the speech in real-time.
Emotion Summary: The chatbot can provide an overview of the emotional tones expressed throughout the conversation.
Advanced Audio and Text Analysis: Combining speech recognition, translation, and emotion detection increases the complexity and human-likeness of the interaction.
This project adds a significant layer of complexity compared to the previous one by incorporating emotion detection and emotionally-aware responses, making the chatbot more empathetic and human-like in its interactions. This would be especially useful in mental health apps, customer support, or emotional AI assistants.
"""