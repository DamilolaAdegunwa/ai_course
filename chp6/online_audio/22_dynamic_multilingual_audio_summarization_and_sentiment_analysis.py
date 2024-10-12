"""
Project Title: Dynamic Multilingual Audio Summarization and Sentiment Analysis for Podcast Episodes
File Name: dynamic_multilingual_audio_summarization_and_sentiment_analysis.py

Project Description:
In this project, we will develop an advanced multilingual audio summarization system for podcast episodes, capable of generating real-time summaries and performing sentiment analysis on various segments. This project will:

Transcribe multilingual podcast audio using OpenAI's Whisper model.
Dynamically summarize sections of the podcast based on configurable segment durations (e.g., every 5 minutes).
Perform sentiment analysis on each summarized segment to classify the tone (positive, neutral, or negative).
Generate a comprehensive summary and overall sentiment analysis of the entire episode, which can be exported as a report for podcast listeners.
This project introduces a higher level of complexity by combining audio transcription, real-time summarization, and sentiment analysis, offering multilingual capabilities and processing long-form content like podcasts.

Python Code:
"""
import os
import librosa
from openai import OpenAI
from apikey import apikey
from textblob import TextBlob  # For sentiment analysis
from typing import Tuple, List

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Step 1: Load and Process Audio File
def load_audio(file_path: str) -> Tuple:
    print(f"Loading audio from: {file_path}")
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    return audio_data, sample_rate, duration


# Step 2: Transcribe Multilingual Audio
def transcribe_audio(file_path: str) -> str:
    print("Transcribing multilingual podcast audio...")

    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    transcribed_text = response['text']
    print("Transcription completed.")
    return transcribed_text


# Step 3: Segment Audio into Chunks for Summarization
def segment_transcription(transcription: str, duration: float, segment_length: float) -> List[str]:
    print(f"Segmenting transcription into {segment_length}-minute intervals...")
    # Segment the transcription into smaller chunks based on duration
    segments = []
    words = transcription.split()
    total_words = len(words)
    words_per_segment = int(total_words * (segment_length / duration))

    for i in range(0, total_words, words_per_segment):
        segment = " ".join(words[i:i + words_per_segment])
        segments.append(segment)

    return segments


# Step 4: Generate Summary for Each Segment
def summarize_segment(segment: str) -> str:
    print("Generating summary for audio segment...")
    prompt = f"Summarize the following podcast segment:\n\n{segment}"

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )

    summary = response['choices'][0]['text'].strip()
    return summary


# Step 5: Perform Sentiment Analysis
def analyze_sentiment(segment: str) -> str:
    print("Performing sentiment analysis...")
    blob = TextBlob(segment)
    sentiment = blob.sentiment.polarity

    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"


# Step 6: Generate Final Podcast Report (Summaries + Sentiment Analysis)
def generate_podcast_report(transcription: str, segment_length: float, duration: float) -> str:
    print("Generating podcast report...")

    segments = segment_transcription(transcription, duration, segment_length)
    report = []

    for idx, segment in enumerate(segments):
        summary = summarize_segment(segment)
        sentiment = analyze_sentiment(segment)
        report.append(f"Segment {idx + 1} Summary: {summary}")
        report.append(f"Sentiment: {sentiment}\n")

    return "\n".join(report)


# Main function to process a podcast audio file
def process_podcast_audio(file_path: str, segment_length: float = 5.0):
    # Step 1: Load audio file and get its duration
    audio_data, sample_rate, duration = load_audio(file_path)

    # Step 2: Transcribe the audio
    transcription = transcribe_audio(file_path)

    # Step 3: Generate podcast report with summaries and sentiment analysis
    podcast_report = generate_podcast_report(transcription, segment_length, duration)

    # Output the report to a file
    report_file = "podcast_summary_report.txt"
    with open(report_file, "w") as file:
        file.write(podcast_report)

    print(f"Podcast summary and sentiment analysis report saved to {report_file}")


# Run the project
if __name__ == "__main__":
    podcast_file_path = "path_to_podcast_file.mp3"  # Provide the path to the podcast audio file
    process_podcast_audio(podcast_file_path, segment_length=5.0)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A 30-minute multilingual podcast episode where hosts discuss global politics and social issues.

Expected Output:

Summarized Segments: A 5-minute breakdown of key points in the conversation.
Sentiment Analysis: Classification of each segment as positive, negative, or neutral based on the tone of the discussion.
Final Report: A file containing summaries of all segments and the overall sentiment of the podcast.
vbnet
Copy code
Segment 1 Summary: The hosts discuss the recent political developments in Europe, focusing on Brexit.
Sentiment: Neutral

Segment 2 Summary: The conversation shifts to the impact of climate change on global economics.
Sentiment: Negative
...
Example 2:

Input Audio: A 20-minute technology podcast where hosts talk about recent innovations in AI.

Expected Output:

Summarized Segments: Every 5 minutes, a concise summary of what was discussed.
Sentiment Analysis: Based on how optimistic or critical the hosts sound regarding AI developments.
Final Report: A summary report with sentiment analysis.
yaml
Copy code
Segment 1 Summary: The hosts highlight the advancements in machine learning algorithms.
Sentiment: Positive

Segment 2 Summary: Discussion shifts towards ethical concerns and bias in AI models.
Sentiment: Negative
...
Key Features:
Multilingual Support: The system can handle audio in different languages, making it suitable for international podcast episodes.
Dynamic Audio Summarization: The podcast is summarized every few minutes, creating shorter, digestible summaries that can be consumed separately.
Sentiment Analysis: Automatically detects the emotional tone of each segment (positive, negative, or neutral), giving a better understanding of the podcast's mood.
Podcast Report Generation: A complete report that includes both summaries and sentiment analysis is produced, which can be used for podcast analytics or for listeners who want quick insights.
Use Cases:
Podcast Analytics: Use this tool to analyze podcasts and generate summaries for listeners who may want to skim through long episodes.
Content Creators: Automatically create summaries and sentiment insights for uploaded podcast content, providing added value to subscribers.
Multilingual Audio Processing: Process podcasts or other audio content in multiple languages, making this tool ideal for international podcasts.
Market Research: Analyze large volumes of audio content, extracting key insights and emotional tone, useful for market research firms monitoring media sentiment.
Educational Tools: For educational podcast platforms, this project can provide both summarizations and sentiment classifications, making it easier for students to study long-form content.
This project introduces an advanced workflow by not only transcribing multilingual podcasts but also summarizing key sections and conducting sentiment analysis on each segment. It’s ideal for content creators and podcast analytics platforms looking to offer enhanced features such as real-time insights into emotional tone and digestible summaries of long-form content.
"""
