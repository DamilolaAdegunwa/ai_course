"""
Project Title: Real-Time Audio-Based Multilingual Speech Translation and Sentiment Response Generation
File Name: real_time_audio_multilingual_speech_translation_and_sentiment_response.py

Project Description:
In this project, you’ll develop a real-time multilingual speech translation system with the ability to:

Transcribe and translate live audio input in real-time from one language to another using OpenAI’s Whisper API.
Detect the sentiment (positive, neutral, or negative) of the translated text.
Generate appropriate responses based on the detected sentiment.
Translate the chatbot’s response back into the original language of the speaker and synthesize it into speech, providing a complete multilingual, real-time interaction.
This project is significantly more complex because it integrates:

Real-time transcription and translation.
Sentiment analysis of multilingual text.
Two-way translation: Both translating the input into English for analysis and translating responses back to the original language.
Real-time speech synthesis in the user's language.
Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
import pyaudio
import pyttsx3
from io import BytesIO

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech (TTS) engine
tts_engine = pyttsx3.init()

# Supported languages for translation
supported_languages = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}


# Function to detect the language of the audio and translate to English
def transcribe_and_translate(audio_chunk, target_language="en"):
    audio_file = BytesIO(audio_chunk)

    # Step 1: Transcribe the audio and detect language
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    transcription = response['text']
    detected_language = response.get('language', 'en')  # Fallback to English if not detected

    # Step 2: Translate to the target language (English for sentiment analysis)
    translation = client.translations.create(
        model="whisper-1",
        source_language=detected_language,
        target_language=target_language,
        text=transcription
    )
    translated_text = translation['translated_text']

    return transcription, translated_text, detected_language


# Function to analyze sentiment of the translated text
def analyze_sentiment(text):
    sentiment_response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the sentiment of this text: '{text}'. Is it positive, neutral, or negative?",
        max_tokens=30
    )
    sentiment = sentiment_response['choices'][0]['text'].strip().lower()

    return sentiment


# Function to generate a response based on sentiment and translate it back to the user's language
def generate_response_and_translate(sentiment, target_language):
    response_generation = client.completions.create(
        model="text-davinci-003",
        prompt=f"Generate a polite and empathetic response to a {sentiment} sentiment statement.",
        max_tokens=100
    )
    response = response_generation['choices'][0]['text'].strip()

    # Translate response back to the user's language
    translation_back = client.translations.create(
        model="whisper-1",
        source_language="en",
        target_language=target_language,
        text=response
    )

    translated_response = translation_back['translated_text']

    return translated_response


# Function to convert text to speech (TTS)
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


# Main function for the real-time multilingual translation and sentiment chatbot
def multilingual_chatbot():
    print("Welcome to the real-time multilingual chatbot!")

    try:
        while True:
            # Step 1: Record user input
            audio_chunk = record_audio_for_duration(duration=5)

            # Step 2: Transcribe and translate the user's input to English
            transcription, translated_text, detected_language = transcribe_and_translate(audio_chunk)

            print(f"\nYou said (in {supported_languages.get(detected_language, 'unknown language')}): {transcription}")
            print(f"Translated to English: {translated_text}")

            # Step 3: Analyze the sentiment of the translated text
            sentiment = analyze_sentiment(translated_text)
            print(f"Detected sentiment: {sentiment}")

            # Step 4: Generate a response based on the sentiment and translate it back to the original language
            translated_response = generate_response_and_translate(sentiment, detected_language)
            print(f"Chatbot response (translated back): {translated_response}")

            # Step 5: Speak the response in the original language
            speak_text(translated_response)

            # Exit the chatbot if the user says "Goodbye"
            if "goodbye" in transcription.lower() or "bye" in transcription.lower():
                print("Goodbye! Ending the conversation.")
                speak_text("Goodbye! Ending the conversation.")
                break

    except Exception as e:
        print(f"Error: {e}")


# Run the multilingual chatbot
if __name__ == "__main__":
    multilingual_chatbot()
"""
Example Inputs and Expected Outputs:
Example 1:

Input (spoken in French): "Je suis tellement fatigué de tout ça."
Expected Output:
Transcription: "Je suis tellement fatigué de tout ça."
Translated to English: "I am so tired of all this."
Sentiment: "Negative"
Chatbot Response (in English): "I'm sorry you're feeling this way. Sometimes it's okay to take a step back and rest."
Translated Response (back to French): "Je suis désolé que tu te sentes ainsi. Parfois, il est bon de faire une pause et de se reposer."
TTS Output: Chatbot speaks: "Je suis désolé que tu te sentes ainsi. Parfois, il est bon de faire une pause et de se reposer."
Example 2:

Input (spoken in Spanish): "Hoy ha sido un día increíble."
Expected Output:
Transcription: "Hoy ha sido un día increíble."
Translated to English: "Today has been an incredible day."
Sentiment: "Positive"
Chatbot Response (in English): "That's fantastic! I'm glad you're having such a great day!"
Translated Response (back to Spanish): "¡Eso es fantástico! ¡Me alegra que estés teniendo un día tan increíble!"
TTS Output: Chatbot speaks: "¡Eso es fantástico! ¡Me alegra que estés teniendo un día tan increíble!"
Key Concepts and Features:
Real-Time Multilingual Transcription: The system transcribes speech in any supported language and translates it to English for processing.
Sentiment Analysis: Based on the translated text, the chatbot detects the sentiment and tailors its response accordingly.
Bi-Directional Translation: After generating a response in English, the system translates it back to the original language for output.
Speech Synthesis in Multiple Languages: The chatbot speaks the final response in the user’s language for a seamless conversation flow.
Exit Condition: The user can terminate the conversation with commands like "Goodbye."
This project is more complex than the previous one because it involves multilingual capabilities, real-time translations, bi-directional sentiment analysis, and speech synthesis, all while maintaining a conversational flow. This system could be a foundation for cross-language customer service bots, multilingual virtual assistants, or translation tools with emotional intelligence.
"""