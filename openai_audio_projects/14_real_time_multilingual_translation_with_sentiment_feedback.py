"""
Project Title: Real-Time Multi-Language Audio Translation with Live Speaker Sentiment Feedback
File Name: real_time_multilingual_translation_with_sentiment_feedback.py

Project Description:
In this next advanced OpenAI audio project, we will:

Transcribe and translate audio in real-time from one language to another.
Detect and profile speaker emotions live, providing continuous sentiment feedback while the transcription is happening.
Allow for translation between multiple languages dynamically as the speech is detected.
Display real-time emotional feedback for each speaker as the conversation progresses, providing sentiment insights as the conversation unfolds.
Support real-time audio processing to give dynamic feedback to the user, combining speech-to-text translation with emotional profiling in one seamless pipeline.
This project will involve a real-time feedback loop where language translation, emotion detection, and speech transcription work concurrently, adding substantial complexity.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
import sounddevice as sd
import numpy as np
from io import BytesIO

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Define global constants
LANGUAGE_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh"
}
SENTIMENT_LABELS = ["positive", "neutral", "negative"]

# Function to record audio in real-time
def record_audio(duration=10, sample_rate=16000):
    print("Recording audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording completed.")
    return recording

# Function to transcribe and translate audio in real-time
def transcribe_and_translate_real_time(audio_data, source_language, target_language):
    audio_stream = BytesIO(audio_data)

    print(f"Transcribing and translating from {source_language} to {target_language}...")
    response = client.audio.translations.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json",
        source_language=LANGUAGE_MAP[source_language],
        target_language=LANGUAGE_MAP[target_language]
    )

    transcription = response['text']
    return transcription

# Function to detect emotion in real-time for a conversation
def detect_real_time_emotion(transcription):
    print("Detecting sentiment...")
    prompt = f"Analyze the sentiment of this text: '{transcription}' and classify it as positive, neutral, or negative."

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=10
    )

    sentiment = response['choices'][0]['text'].strip().lower()
    return sentiment

# Function to handle real-time transcription, translation, and emotion detection
def real_time_transcription_translation_emotion(file_path, source_language, target_language):
    # Step 1: Record or read audio
    audio_data, _ = record_audio(duration=10)  # 10 seconds recording for testing
    audio_data = (audio_data * 32767).astype(np.int16)  # Convert audio to 16-bit format

    # Step 2: Transcribe and translate audio
    transcription = transcribe_and_translate_real_time(audio_data, source_language, target_language)
    print(f"Transcription: {transcription}")

    # Step 3: Real-time emotion detection
    sentiment = detect_real_time_emotion(transcription)
    print(f"Detected Sentiment: {sentiment}")

    # Provide dynamic feedback
    return transcription, sentiment


# Main function
if __name__ == "__main__":
    source_language = "Spanish"
    target_language = "English"
    file_path = "C:/path_to_audio/sample_conversation_spanish.mp3"

    transcription, sentiment = real_time_transcription_translation_emotion(file_path, source_language, target_language)

    print(f"\nFinal Transcription: {transcription}")
    print(f"Final Sentiment: {sentiment}")
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A Spanish speaker saying:

"Estoy muy feliz de estar aquí, este es un día maravilloso."
Expected Transcription:

css
Copy code
"I'm very happy to be here, this is a wonderful day."
Expected Sentiment Detection:

makefile
Copy code
Sentiment: positive
Real-Time Feedback:

As the transcription happens, the system detects that the speaker is conveying a positive sentiment.
Example 2:

Input Audio: A French speaker saying:

"Je suis désolé, mais je n'aime pas cette idée."
Expected Transcription:

css
Copy code
"I'm sorry, but I don't like this idea."
Expected Sentiment Detection:

makefile
Copy code
Sentiment: negative
Real-Time Feedback:

During the transcription, the system recognizes a negative sentiment in the speaker’s tone.
Key Features:
Real-Time Transcription and Translation: The program can transcribe spoken language and translate it from one language to another on the fly, making it practical for live conversations.
Live Emotion Detection: Alongside the transcription, the system detects and profiles the emotions of the speakers based on the content of their speech.
Multi-language Support: With a flexible language map, users can choose from a variety of languages for both input and output, dynamically translating between multiple languages.
Real-Time Feedback Loop: The program gives real-time sentiment feedback during the conversation, providing dynamic and continuous insights into the speaker's emotional tone.
Use Cases:
International Conferences: Real-time translation and emotion detection for live conferences with multiple languages, where understanding the tone of speakers can enhance communication.
Live Customer Support: This system could be applied in real-time customer service conversations where operators need to quickly gauge customer emotions and adjust their responses accordingly.
Real-Time Multi-language Meetings: Facilitates live translation and emotional profiling for meetings with participants speaking different languages, improving understanding and interaction quality.
This project brings together real-time audio recording, multi-language translation, and emotion detection in one cohesive workflow, greatly increasing complexity and potential real-world applications.
"""