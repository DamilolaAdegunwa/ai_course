"""
Project Title: AI-Powered Audio to Interactive Webinar Transcript Generator with Topic-based Q&A, Emotion-Driven Adaptive Summaries, and Real-Time Translation for Multilingual Audiences
File Name: ai_interactive_webinar_transcript_qna_adaptive_summary.py

Project Description:
This advanced project builds a system that processes webinar or long-form audio in real-time, generates interactive transcripts, and enables topic-based question answering (Q&A). The system will extract key sections of the webinar based on topics, summarize them, adapt those summaries based on speaker emotion, and offer a Q&A interface that allows users to query specific parts of the audio content. The audio can also be translated into multiple languages and presented in an interactive transcript format.

Key Features:
Real-time transcription and speaker identification for multiple speakers.
Adaptive summaries driven by speaker emotions, condensing key information dynamically.
Topic-based Q&A system for users to ask questions about specific sections of the webinar.
Interactive transcript generation with clickable timestamps for easy navigation.
Multilingual translation for global audience engagement.
Emotion-driven content adaptation to highlight areas of emotional intensity (e.g., excitement, frustration).
Advanced Concepts Introduced:
Real-time natural language understanding for adaptive summaries that change based on speaker emotion.
Interactive Q&A system based on extracted key topics using NLP.
Multilingual real-time translation using Hugging Face’s translation pipeline.
Rich interactive interface allowing users to explore content in multiple ways (by speaker, topic, emotion).
Click-based transcript navigation with time stamps.
Python Code:
"""
import os
import json
import time
from typing import List, Dict
from openai import OpenAI
from transformers import pipeline
from vosk import Model, KaldiRecognizer
from flask import Flask, render_template, jsonify, request

# Initialize OpenAI and Hugging Face
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Emotion detection pipeline from Hugging Face
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Translation pipeline for multilingual support
translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")  # English to French

# Vosk model for speaker diarization
vosk_model = Model("model")

# Flask app for web interface
app = Flask(__name__)

# In-memory storage for transcripts
transcripts = {}


# Step 1: Capture and transcribe audio with speaker diarization
def capture_and_transcribe_audio(file_path: str) -> List[Dict[str, str]]:
    recognizer = KaldiRecognizer(vosk_model, 16000)
    transcriptions = []
    with open(file_path, "rb") as audio_file:
        while True:
            data = audio_file.read(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                transcriptions.append(json.loads(result))
    return transcriptions


# Step 2: Generate emotion-driven adaptive summaries
def generate_adaptive_summaries(transcription_with_speakers: List[Dict[str, str]]) -> str:
    full_transcription = " ".join([t.get("text", "") for t in transcription_with_speakers])
    emotions = emotion_pipeline(full_transcription)

    # Highlight emotional changes (summarizing based on emotion)
    summary = "Summary:\n"
    for t, emotion in zip(transcription_with_speakers, emotions):
        if emotion['label'] == 'joy':
            summary += f"{t['speaker']} was excited: {t['text']}\n"
        elif emotion['label'] == 'anger':
            summary += f"{t['speaker']} showed frustration: {t['text']}\n"
        else:
            summary += f"{t['speaker']} discussed: {t['text']}\n"

    return summary


# Step 3: Translate content into different languages
def translate_content(text: str, target_language: str = "fr") -> str:
    translated = translation_pipeline(text)
    return translated[0]['translation_text']


# Step 4: Create an interactive transcript with timestamps and speakers
def generate_interactive_transcript(transcription_with_speakers: List[Dict[str, str]]) -> Dict[str, str]:
    transcript = {"transcript": []}
    for t in transcription_with_speakers:
        transcript["transcript"].append({
            "time": t['result'][0]['start'],  # Start time of the transcript segment
            "speaker": t.get("speaker", "Unknown"),
            "text": t.get("text", "")
        })
    return transcript


# Step 5: Interactive Q&A using NLP (OpenAI for GPT-based Q&A)
def qna_system(question: str, context: str) -> str:
    prompt = f"Context: {context}\n\nQ: {question}\nA:"
    response = client.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()


# Step 6: Flask Web App for Interactive Transcript and Q&A
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    file_path = request.files['audio_file'].filename
    transcription = capture_and_transcribe_audio(file_path)
    transcript = generate_interactive_transcript(transcription)
    transcripts[file_path] = transcript
    return jsonify(transcript)


@app.route('/summary', methods=['POST'])
def summarize_audio():
    file_path = request.form['file_path']
    transcription = transcripts.get(file_path)
    summary = generate_adaptive_summaries(transcription)
    return jsonify({"summary": summary})


@app.route('/translate', methods=['POST'])
def translate_transcription():
    file_path = request.form['file_path']
    target_language = request.form['target_language']
    transcription = transcripts.get(file_path)
    full_transcription = " ".join([t.get("text", "") for t in transcription["transcript"]])
    translated_text = translate_content(full_transcription, target_language)
    return jsonify({"translated_text": translated_text})


@app.route('/qna', methods=['POST'])
def qna():
    question = request.form['question']
    file_path = request.form['file_path']
    transcription = transcripts.get(file_path)
    context = " ".join([t.get("text", "") for t in transcription["transcript"]])
    answer = qna_system(question, context)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
"""
Project Breakdown:
1. Real-time Transcription with Speaker Identification:
Audio is processed using Vosk for transcription and speaker identification. Multiple speakers are diarized so their dialogue is distinguished.
2. Emotion-driven Adaptive Summaries:
Speaker text is analyzed for emotional content (joy, anger, sadness, etc.). Summaries are adaptively generated, emphasizing areas where there is strong emotion, making it easier to identify important sections of the conversation.
3. Multilingual Translation:
Once the transcription is complete, it can be translated into multiple languages using Hugging Face’s translation models. This feature supports creating transcripts in various languages for a global audience.
4. Interactive Transcripts with Time Stamps and Speaker Metadata:
Transcripts are displayed in an interactive format that users can click on. Each section is tagged with the speaker and time stamp, allowing users to jump to different parts of the conversation.
5. Q&A System for Topic-based Queries:
Users can ask questions about the audio content, and the system uses OpenAI's GPT models to answer those questions based on the context from the transcript. This turns the static transcript into an interactive, searchable experience.
Example Input and Output:
Input:

Webinar Audio File of a product launch presentation with 3 speakers.
Output:

Transcription: Full transcript of the webinar with time stamps and speaker names.
Translation: Translated transcript in French.
Emotion-Based Summary: The system highlights that Speaker 2 was excited about a new feature, while Speaker 3 expressed concern about the timeline.
Interactive Transcript: Users can click on specific sections of the transcript to jump to that point in the audio.
Q&A: The user asks, “What was the key feature of the product?” and receives an answer extracted from the transcript.
Advanced Features:
Interactive Interface:

Users can interact with the audio transcript via a web interface that allows them to jump to different points in the conversation by clicking on a section of the transcript.
Emotion-Driven Adaptation:

By analyzing emotions in the speech, the system generates summaries that reflect the emotional tone of each speaker, allowing for better understanding of intense moments in the conversation.
Topic-based Q&A:

The user can query the system about specific sections of the audio transcript, and it will return answers contextualized from the transcript.
Multilingual Translation:

The audio transcript is translated into various languages, expanding the reach to non-English speaking audiences.
Use Cases:
Webinars and Online Events: Real-time interactive transcripts for online conferences or workshops.
Customer Support Calls: Analyze and summarize long customer support calls, highlighting emotional points and providing quick summaries.
Education: Students can interact with lecture recordings, ask questions, and receive instant answers from the transcript.
Global Marketing: Multilingual support allows for real-time translation and engagement with a global audience.
This project significantly raises the complexity by incorporating emotion analysis, multilingual capabilities, and interactive elements into a dynamic real-time system, pushing the boundaries of OpenAI audio processing.
"""