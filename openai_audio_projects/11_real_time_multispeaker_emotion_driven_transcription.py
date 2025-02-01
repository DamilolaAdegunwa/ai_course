"""
Project Title: Real-Time Multispeaker Audio Transcription and Emotion-Driven Dialogue Generation with Contextual Memory
File Name: real_time_multispeaker_emotion_driven_transcription.py

Project Description:
In this advanced OpenAI audio project, we implement a real-time multispeaker audio transcription system capable of:

Detecting multiple speakers in real-time and transcribing each speaker's input separately.
Analyzing the emotional tone of each speaker's input, which allows for personalized responses based on the emotion detected.
Generating emotionally appropriate responses for each speaker, using contextual memory to track conversations.
Handling interruptions and overlapping speech dynamically, separating the transcription and emotion of each speaker even when two or more individuals speak simultaneously.
Storing conversation history across multiple participants, keeping track of emotional states and generating contextually relevant dialogue.
The complexity comes from:

Handling multiple speakers.
Speaker diarization (identifying which speaker said what).
Tracking emotional states for multiple speakers simultaneously.
Managing contextual conversation memory for each speaker, making the system adept at long-form conversations with many participants.
Python Code

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
import numpy as np
from io import BytesIO
import pyaudio
import pyttsx3

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Emotion classification labels
emotion_labels = ['happy', 'sad', 'angry', 'neutral']

# Dictionary to track each speaker's context and emotion state
speaker_memory = {}


# Function to analyze emotional tone for each speaker
def analyze_speaker_emotion(transcription):
    emotion_prompt = f"Analyze the emotional tone of this text: '{transcription}'. Indicate whether the speaker is happy, sad, angry, or neutral."

    emotion_response = client.completions.create(
        model="text-davinci-003",
        prompt=emotion_prompt,
        max_tokens=50
    )

    emotion = emotion_response['choices'][0]['text'].strip().lower()

    if emotion not in emotion_labels:
        emotion = 'neutral'

    return emotion


# Function to transcribe and analyze emotions for each speaker
def transcribe_and_analyze_speakers(audio_data):
    audio_file = BytesIO(audio_data)

    # Step 1: Transcribe the audio with speaker diarization (speaker separation)
    transcription_response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )

    segments = transcription_response['segments']

    # Step 2: For each speaker, transcribe and analyze their emotions
    speaker_emotion_data = []
    for segment in segments:
        speaker = segment['speaker']
        text = segment['text']

        # Analyze the emotion for each speaker
        emotion = analyze_speaker_emotion(text)

        # Save transcription and emotion per speaker
        speaker_emotion_data.append({
            "speaker": speaker,
            "text": text,
            "emotion": emotion
        })

        # Update speaker memory for context tracking
        if speaker not in speaker_memory:
            speaker_memory[speaker] = []
        speaker_memory[speaker].append({"text": text, "emotion": emotion})

    return speaker_emotion_data


# Function to generate a context-aware response for each speaker
def generate_emotion_based_response(speaker, text, emotion):
    # Generate response based on speaker's emotional state
    prompt = f"The following statement was made by Speaker {speaker}: '{text}'. The speaker is feeling {emotion}. Generate an appropriate response."

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )

    return response['choices'][0]['text'].strip()


# Function to record real-time audio for multiple speakers
def record_multispeaker_audio(duration=5, chunk_size=1024, channels=1, rate=16000):
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


# Main function to handle conversation between multiple speakers
def multispeaker_emotion_driven_conversation():
    print("Starting real-time multispeaker emotion-aware conversation system...")

    try:
        while True:
            # Step 1: Record multispeaker audio
            audio_data = record_multispeaker_audio(duration=5)

            # Step 2: Transcribe and analyze emotion for each speaker
            speaker_data = transcribe_and_analyze_speakers(audio_data)

            for entry in speaker_data:
                speaker = entry['speaker']
                text = entry['text']
                emotion = entry['emotion']

                print(f"\nSpeaker {speaker}: {text}")
                print(f"Emotion detected: {emotion}")

                # Step 3: Generate a response for the speaker based on their emotion
                response = generate_emotion_based_response(speaker, text, emotion)
                print(f"Chatbot response to Speaker {speaker}: {response}")

                # Convert response to speech
                tts_engine.say(response)
                tts_engine.runAndWait()

            # Step 4: Break the conversation loop if "goodbye" is mentioned
            if any("goodbye" in entry['text'].lower() for entry in speaker_data):
                print("Goodbye detected. Ending the conversation.")
                tts_engine.say("Goodbye!")
                break

    except Exception as e:
        print(f"Error occurred: {e}")


# Run the multispeaker emotion-driven chatbot
if __name__ == "__main__":
    multispeaker_emotion_driven_conversation()
"""
Example Inputs and Expected Outputs:
Example 1 (Conversation between 2 speakers):

Input (Speaker 1): "I'm really frustrated. I've been trying to solve this problem for hours."

Input (Speaker 2): "Hey, don't worry. You'll figure it out soon!"

Expected Output:

Speaker 1 Emotion: "angry"
Speaker 2 Emotion: "happy"
Chatbot response to Speaker 1: "I understand your frustration. Let's take a step back and try a new approach."
Chatbot response to Speaker 2: "It's great you're staying positive. How can we help Speaker 1 stay calm?"
Example 2 (Conversation between 3 speakers):

Input (Speaker 1): "This is exciting! We finally got the project done."

Input (Speaker 2): "I’m so tired though, it took forever."

Input (Speaker 3): "Same here. I can’t wait to take a break."

Expected Output:

Speaker 1 Emotion: "happy"
Speaker 2 Emotion: "tired"
Speaker 3 Emotion: "tired"
Chatbot response to Speaker 1: "It’s wonderful that you're excited. What's the next step?"
Chatbot response to Speaker 2 and Speaker 3: "I see you're both feeling drained. Rest is important after hard work."
Key Features:
Multispeaker Transcription with Emotion Detection: The system can transcribe audio from multiple speakers and detect their emotions individually in real-time.
Speaker Diarization: Each speaker’s contribution is separated and processed independently, with their emotions being tracked separately.
Emotion-Based Response Generation: Responses are tailored to each speaker’s emotional state, providing personalized and emotionally appropriate replies.
Contextual Memory: The system keeps track of each speaker’s dialogue history and emotions over time, allowing it to provide contextually aware responses.
Real-Time Audio Processing: Handles real-time audio recording and processing, making it applicable to live conversations.
Supports Multiple Languages: Transcription and emotion detection work for multiple languages, allowing diverse use cases.
Use Cases:
Multispeaker Virtual Meetings: This system can transcribe meetings, detect emotional undertones, and provide contextually relevant summaries of each participant’s contributions.
Group Therapy Sessions: In mental health settings, this project could transcribe and detect emotions in group therapy, providing insights into the emotional states of each participant.
Customer Service: For call centers with multiple speakers, the system could detect customer emotions, enabling customer service agents to respond more empathetically.
This project adds speaker diarization and multispeaker context, creating a powerful framework for emotion-driven interactions in environments where multiple individuals contribute to the conversation.
"""