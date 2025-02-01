"""
Project Title: AI-Powered Podcast Summarization with Key Topic Extraction and Voice Modulation
File Name: podcast_summarization_topic_extraction_voice_modulation.py

Project Description:
In this advanced project, we will create a system that:

Processes long audio files (such as podcasts or interviews) using OpenAI's Whisper model for transcription.
Summarizes the entire podcast, highlighting key points and breaking down the content into digestible sections.
Performs topic extraction to identify the main subjects discussed throughout the podcast.
Modulates the voice of the speaker dynamically based on the topics being discussed (e.g., adding a deeper, authoritative tone for serious topics or a lighter, enthusiastic tone for inspirational topics).
Provides a full report that includes the summary, the main topics discussed, timestamps for each topic, and any voice modulation applied.
This project combines summarization, topic modeling, and audio processing through voice modulation. It requires building a more complex pipeline that deals with both the semantic content of the audio and how it's presented in terms of voice.

Python Code:
"""
import os
import librosa
import numpy as np
from transformers import pipeline
from openai import OpenAI
from apikey import apikey
from gensim.summarization import summarize
from pydub import AudioSegment
from scipy.io.wavfile import write
from typing import Tuple, List

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Step 1: Transcribe the Podcast
def transcribe_podcast(file_path: str) -> str:
    print("Transcribing podcast...")
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    transcription = response['text']
    return transcription


# Step 2: Summarize Transcription
def summarize_transcription(transcription: str) -> str:
    print("Summarizing podcast transcription...")
    summary = summarize(transcription, ratio=0.1)  # Summarize to 10% of original text
    return summary


# Step 3: Extract Key Topics
def extract_topics(transcription: str) -> List[Tuple[str, float]]:
    print("Extracting key topics from the transcription...")
    nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    topics = ["Technology", "Sports", "Politics", "Science", "Health", "Education", "Entertainment"]

    result = nlp(transcription, candidate_labels=topics)
    extracted_topics = [(label, score) for label, score in zip(result['labels'], result['scores'])]

    return extracted_topics


# Step 4: Modulate Voice Based on Topic
def modulate_voice(audio_path: str, topic: str) -> str:
    print(f"Modulating voice for topic: {topic}")
    audio = AudioSegment.from_file(audio_path)

    if topic == "Politics" or topic == "Science":
        # Deeper, more authoritative voice
        modulated_audio = audio.low_pass_filter(300).speedup(0.9)
    elif topic == "Entertainment" or "Education":
        # Lighter, more enthusiastic tone
        modulated_audio = audio.high_pass_filter(400).speedup(1.1)
    else:
        # Default modulation for neutral topics
        modulated_audio = audio.speedup(1.0)

    # Save modulated audio
    modulated_file_path = f"modulated_{topic}.wav"
    modulated_audio.export(modulated_file_path, format="wav")

    return modulated_file_path


# Step 5: Generate Podcast Report
def generate_podcast_report(transcription: str, summary: str, topics: List[Tuple[str, float]]):
    print("Generating podcast report...")
    report = f"Podcast Summary:\n{summary}\n\nKey Topics and Scores:\n"
    for topic, score in topics:
        report += f"Topic: {topic}, Score: {score:.2f}\n"

    with open("podcast_report.txt", "w") as report_file:
        report_file.write(report)

    print("Podcast report saved to 'podcast_report.txt'.")


# Step 6: Full Processing Pipeline
def process_podcast(file_path: str):
    # Step 1: Transcribe the podcast
    transcription = transcribe_podcast(file_path)

    # Step 2: Summarize the transcription
    summary = summarize_transcription(transcription)

    # Step 3: Extract topics
    topics = extract_topics(transcription)

    # Step 4: Modulate voice for each topic
    for topic, score in topics:
        modulated_audio_path = modulate_voice(file_path, topic)
        print(f"Modulated audio saved for topic '{topic}' at: {modulated_audio_path}")

    # Step 5: Generate podcast report
    generate_podcast_report(transcription, summary, topics)


if __name__ == "__main__":
    # Path to the podcast file
    podcast_file = "example_podcast.wav"

    # Process the podcast
    process_podcast(podcast_file)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A podcast discussing technology, education, and politics.

Expected Output:

Original Transcription: The entire content of the podcast.
Summary: "The podcast discusses the latest advancements in technology, particularly AI, and its application in education. The speakers also delve into political challenges regarding AI ethics."
Topics Extracted:
Technology (0.89)
Education (0.82)
Politics (0.76)
Modulated Audio:
Audio segments are saved as modulated_Technology.wav, modulated_Education.wav, and modulated_Politics.wav.
The Technology segment has a deeper, authoritative tone.
The Education segment has a lighter, faster tone.
Example 2:

Input Audio: A podcast with a focus on health, science, and entertainment.

Expected Output:

Original Transcription: The podcast covers a variety of health tips, recent scientific discoveries, and entertainment news.
Summary: "In this episode, we cover health tips for improving mental well-being, breakthroughs in vaccine research, and highlights from the entertainment industry."
Topics Extracted:
Health (0.88)
Science (0.85)
Entertainment (0.78)
Modulated Audio:
Audio segments are saved as modulated_Health.wav, modulated_Science.wav, and modulated_Entertainment.wav.
The Health and Science segments are modulated to sound deeper and slower.
The Entertainment segment has a higher pitch and faster speed.
Key Features:
Advanced Transcription: Uses OpenAI's Whisper model to transcribe long audio files.
Summarization: Uses Gensim's summarization to generate a concise summary of the podcast.
Topic Extraction: Identifies the main topics in the transcription using a zero-shot classification model from Hugging Face's transformers.
Voice Modulation: Changes the audio characteristics (pitch, speed, filter) based on the topics being discussed, adding auditory enhancements.
Podcast Report: A detailed report is generated containing the summary, extracted topics, and their relevance scores, saved as a text file.
Use Cases:
Podcast Creators: Automatically generate summaries and enhanced audio based on topics to make content more engaging for listeners.
Educational Institutions: Use voice modulation for different topics in lectures to help keep students more engaged and highlight important segments.
Entertainment Industry: Dynamically adjust audio for interviews, panel discussions, or podcasts to highlight emotions and key subjects.
Business Meetings: Summarize long meetings and modulate the voices of speakers based on the context to make post-meeting reviews more accessible and engaging.
This project is noticeably more complex due to the combination of transcription, summarization, topic extraction, and audio modulation. Each component is handled separately but integrated into a cohesive pipeline that outputs not only a transformed audio file but also a textual analysis. The project challenges you with new elements such as topic modeling, text processing, and dynamic audio manipulation, making it a significant step forward from previous tasks.
"""