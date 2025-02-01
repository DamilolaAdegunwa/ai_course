"""
Project Title: Real-Time Multilingual Audio Transcription and Sentiment Analysis with Whisper
File Name: real_time_multilingual_audio_transcription_sentiment_analysis.py

Project Description:
In this advanced project, you'll create a system that not only transcribes multilingual audio in real time but also performs sentiment analysis on each spoken segment. This combines the Whisper transcription model with OpenAI's language model capabilities to determine the emotional tone of the transcribed text.

By implementing this project, you will enhance your understanding of both audio transcription and text analysis, expanding the scope of your OpenAI audio projects. The system is designed to handle streaming audio inputs (e.g., from a live conference or podcast), transcribe them on-the-fly, and then classify the sentiment of each segment (e.g., positive, negative, neutral).

This project is noticeably more complex due to:

Real-time processing of audio files.
Combining transcription with sentiment analysis.
Breaking down audio into smaller chunks for real-time analysis.
Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey, filepath
import time
import pyaudio  # For real-time audio input
import wave
from io import BytesIO
from typing import BinaryIO

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()
file_path = filepath

# Real-time audio transcription and sentiment analysis function
def transcribe_and_analyze_sentiment(audio_chunk: BinaryIO):
    # Step 1: Transcribe the audio chunk using Whisper
    #audio_file = BytesIO(audio_chunk)
    #audio = open(file_path, "rb")
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_chunk,
        response_format="json"
    )
    print('the response : ')
    print(response)
    transcription = response.text
    print('print the transcription in the transcribe_and_analyze_sentiment:' + transcription)
    # Step 2: Perform sentiment analysis on the transcription
    sentiment_response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        #model="babbage-002",
        #model="davinci-002",
        #model="tts-1",
        prompt=f"Analyze the sentiment of the following text: '{transcription}'.\nReturn either Positive, Negative, or Neutral.",
        max_tokens=100
    )
    print('here is sentiment_response')
    print(sentiment_response)
    sentiment = sentiment_response.choices[0].text.strip()
    print('the sentiment in the transcribe_and_analyze_sentiment: ' + sentiment)
    return transcription, sentiment


# Audio recording setup for real-time input (redundant!)
def record_audio_for_duration(duration=10, chunk_size=1024, channels=1, rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

    print(f"Recording audio for {duration} seconds...")
    frames = []

    # Collect real-time audio for the specified duration
    for _ in range(0, int(rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)

# Main function to run real-time transcription and sentiment analysis
def real_time_transcription_and_sentiment_analysis():
    try:
        # Record a short audio segment (10 seconds)
        #audio_chunk = record_audio_for_duration(duration=10)
        audio_chunk = open(file_path, "rb")
        print('print the audio_chunk in the real_time_transcription_and_sentiment_analysis: ' + file_path)
        # Transcribe and analyze sentiment for the audio chunk
        transcription, sentiment = transcribe_and_analyze_sentiment(audio_chunk)

        # Output the transcription and sentiment analysis
        print(f"\nTranscription: {transcription}")
        print(f"Sentiment: {sentiment}")

    except Exception as e:
        print(f"Error: {e}")


# Running the real-time audio transcription and sentiment analysis
if __name__ == "__main__":
    real_time_transcription_and_sentiment_analysis()
"""
Example Inputs and Expected Outputs:
Example 1:

Input: You record a short, 10-second audio where you say: "I am really excited about the possibilities of artificial intelligence!"
Expected Output:
makefile
Copy code
Transcription: I am really excited about the possibilities of artificial intelligence!
Sentiment: Positive
Example 2:

Input: You record a 10-second audio where you say: "I'm not sure if we should trust AI to handle everything. It could be dangerous."
Expected Output:
vbnet
Copy code
Transcription: I'm not sure if we should trust AI to handle everything. It could be dangerous.
Sentiment: Negative
Key Concepts and Features:
Real-Time Audio Transcription: The system captures and processes audio in real-time using pyaudio for microphone input.
Multilingual Capabilities: Whisper automatically handles multilingual transcription.
Sentiment Analysis Integration: Each segment of the transcribed audio is passed to OpenAI's language model to determine its sentiment (positive, negative, or neutral).
Streaming Capability: The system can be easily extended to process longer or continuous audio streams.
This project builds on your previous experience by incorporating real-time audio processing and adding sentiment analysis, significantly increasing both the complexity and the potential applications.
"""