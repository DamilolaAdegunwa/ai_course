"""
Project Title: Audio Emotion Detection and Content Summarization using Whisper and GPT
File Name: audio_emotion_detection_and_content_summarization.py

Project Description:
In this project, you’ll create a system that transcribes audio, detects the underlying emotions expressed by the speaker in different segments of the audio, and then generates a content summary of the entire audio. This project combines OpenAI's Whisper model for transcription, emotion detection using a fine-tuned GPT model, and text summarization.

The system will:

Transcribe audio in real-time or from a file.
Detect emotions expressed in the audio at different points (e.g., happy, sad, angry, neutral).
Generate a summary of the content, including the emotions identified.
This is a step up from the previous projects by introducing emotion classification at a segment level, and automated content summarization to provide an overview of what was discussed in the audio.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey, filepath
import pyaudio
import wave
from io import BytesIO
from typing import BinaryIO
# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()
file_path = filepath
# Function to transcribe audio and detect emotions
def transcribe_and_detect_emotions(audio_chunk: BinaryIO):
    # Step 1: Transcribe the audio using Whisper
    #audio_file = BytesIO(audio_chunk)
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_chunk,
        response_format="json"
    )
    transcription = response.text

    # Step 2: Emotion detection for each segment of the transcription
    emotion_analysis = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Analyze the following transcription and detect the emotions in it: '{transcription}'.\nReturn emotions for each sentence (e.g., Happy, Sad, Angry, Neutral).",
        max_tokens=150
    )
    emotions = emotion_analysis.choices[0].text.strip()

    return transcription, emotions

# Function to summarize the audio content
def summarize_audio_content(transcription):
    summary_response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Summarize the following transcription: '{transcription}'.",
        max_tokens=100
    )
    summary = summary_response.choices[0].text.strip()

    return summary

# Real-time audio recording
def record_audio_for_duration(duration=10, chunk_size=1024, channels=1, rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

    print(f"Recording audio for {duration} seconds...")
    frames = []

    for _ in range(0, int(rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(frames)

# Main function to transcribe, detect emotions, and summarize content
def audio_emotion_detection_and_summarization():
    try:
        # Step 1: Record audio
        # audio_chunk = record_audio_for_duration(duration=10)
        audio_chunk = open(file_path, "rb")
        # Step 2: Transcribe and detect emotions
        transcription, emotions = transcribe_and_detect_emotions(audio_chunk)

        # Step 3: Summarize the transcription
        summary = summarize_audio_content(transcription)

        # Output results
        print("\nTranscription:\n", transcription)
        print("\nEmotions detected:\n", emotions)
        print("\nSummary of content:\n", summary)

    except Exception as e:
        print(f"Error: {e}")

# Running the emotion detection and content summarization
if __name__ == "__main__":
    audio_emotion_detection_and_summarization()
"""
Example Inputs and Expected Outputs:
Example 1:

Input: Record a 10-second audio where you say: "I’m really happy today because I got a promotion at work! But earlier, I was feeling sad because I missed an important event."
Expected Output:
vbnet
Copy code
Transcription:
I’m really happy today because I got a promotion at work! But earlier, I was feeling sad because I missed an important event.

Emotions detected:
1. "I’m really happy today because I got a promotion at work!" - Happy
2. "But earlier, I was feeling sad because I missed an important event." - Sad

Summary of content:
The speaker expressed happiness due to a promotion but also shared sadness about missing an important event.
Example 2:

Input: Record a 10-second audio where you say: "I'm frustrated because my project isn’t going as planned. But I'm hopeful that things will get better."
Expected Output:
vbnet
Copy code
Transcription:
I'm frustrated because my project isn’t going as planned. But I'm hopeful that things will get better.

Emotions detected:
1. "I'm frustrated because my project isn’t going as planned." - Frustrated
2. "But I'm hopeful that things will get better." - Hopeful

Summary of content:
The speaker is frustrated with a project but remains hopeful for improvement.
Key Concepts and Features:
Emotion Detection: Automatically classify emotions (happy, sad, angry, neutral, hopeful, frustrated, etc.) within the audio transcription.
Content Summarization: Provide a concise summary of the audio's content based on the transcription.
Real-Time Audio Recording and Processing: Capture audio in real-time or from a pre-recorded file, allowing for interactive applications.
Multilingual Support: Whisper can transcribe audio in multiple languages, and the emotion detection and summarization process is language-agnostic.
This project is more advanced than previous ones due to the integration of emotion detection and summarization, two sophisticated natural language processing tasks. This setup can be used for podcasts, interviews, or even customer service scenarios where understanding emotional tone and summarizing content are valuable features.
"""