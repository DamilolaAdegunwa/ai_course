"""
Project Title: Real-Time Conversational Agent with Multi-Tiered Emotional Reactions
File Name: real_time_conversational_agent_with_multi_tiered_emotional_reactions.py

Project Description:
In this advanced project, we develop a real-time conversational agent that listens, transcribes speech, analyzes emotions, and responds with dynamically generated audio reactions based on detected emotional states and conversational flow. The agent reacts not only to the general emotion of the conversation but also responds to specific emotional cues at different stages of the interaction. The system uses OpenAI's Whisper-1 for transcription, GPT-4 for conversation analysis and response generation, and generates corresponding audio reactions using a text-to-speech model. It also uses emotional intensity to influence response tone and speed.

This project also introduces a memory mechanism for keeping track of emotional states over time and adjusting the agent’s reaction accordingly, simulating human-like emotional progression throughout the conversation.

Python Code:
"""
import os
import numpy as np
import sounddevice as sd
from io import BytesIO
from pydub import AudioSegment, playback
from openai import OpenAI
from apikey import apikey
import pyttsx3

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS engine for dynamic audio responses
tts_engine = pyttsx3.init()

# Emotion and Response Memory
EMOTION_MEMORY = []


# Function to record audio in real time
def record_audio(duration=10, sample_rate=16000):
    print("Recording audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording completed.")
    return recording


# Function to transcribe audio and analyze conversation flow
def transcribe_and_analyze_conversation(audio_data):
    audio_stream = BytesIO(audio_data)

    print("Transcribing audio...")
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )

    transcription = response['text']
    print(f"Transcription: {transcription}")

    conversation_analysis_prompt = f"Analyze the following transcription for emotion and conversational flow: '{transcription}'"
    analysis_response = client.completions.create(
        model="gpt-4",
        prompt=conversation_analysis_prompt,
        max_tokens=100
    )

    conversation_analysis = analysis_response['choices'][0]['text'].strip()
    print(f"Conversation Analysis: {conversation_analysis}")
    return transcription, conversation_analysis


# Function to generate dynamic audio response based on emotion and conversational analysis
def generate_audio_response(conversation_analysis):
    # Generate text-based response using GPT-4
    print("Generating conversational response...")
    response_prompt = f"Generate a conversational agent response based on the following emotional analysis: '{conversation_analysis}'"
    response_text = client.completions.create(
        model="gpt-4",
        prompt=response_prompt,
        max_tokens=100
    )['choices'][0]['text'].strip()

    print(f"Agent Response: {response_text}")

    # Analyze emotional tone and adjust TTS parameters
    if 'happy' in conversation_analysis:
        tts_engine.setProperty('rate', 200)  # Faster for happy emotions
        tts_engine.setProperty('volume', 1.0)  # Louder for positive emotions
    elif 'sad' in conversation_analysis:
        tts_engine.setProperty('rate', 150)  # Slower for sad emotions
        tts_engine.setProperty('volume', 0.7)  # Softer volume for sadness
    elif 'angry' in conversation_analysis:
        tts_engine.setProperty('rate', 180)  # Slightly fast for angry emotions
        tts_engine.setProperty('volume', 1.0)  # Louder for anger
    else:
        tts_engine.setProperty('rate', 170)  # Neutral tone
        tts_engine.setProperty('volume', 0.9)

    # Use TTS engine to generate response audio
    tts_engine.say(response_text)
    tts_engine.runAndWait()


# Emotion memory and escalation
def emotion_memory_and_escalation(conversation_analysis):
    global EMOTION_MEMORY

    # Check for emotional change or escalation
    if any(emotion in conversation_analysis for emotion in ['angry', 'sad']):
        EMOTION_MEMORY.append(conversation_analysis)

    if len(EMOTION_MEMORY) >= 3:
        # Escalate if 3 emotional reactions in a row
        escalate_prompt = f"Escalate the agent's response based on the following emotional history: {EMOTION_MEMORY}"
        escalation_response = client.completions.create(
            model="gpt-4",
            prompt=escalate_prompt,
            max_tokens=100
        )['choices'][0]['text'].strip()

        print(f"Escalation Response: {escalation_response}")
        tts_engine.say(escalation_response)
        tts_engine.runAndWait()
        EMOTION_MEMORY.clear()  # Reset after escalation


# Main function for real-time conversational interaction
def real_time_conversational_agent(duration=10):
    # Step 1: Record audio input from user
    audio_data = record_audio(duration=duration)

    # Step 2: Convert numpy array to PCM bytes for transcription and conversation analysis
    audio_data_bytes = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
    transcription, conversation_analysis = transcribe_and_analyze_conversation(audio_data_bytes.tobytes())

    # Step 3: Generate dynamic agent response based on conversational flow and emotion analysis
    generate_audio_response(conversation_analysis)

    # Step 4: Track emotion memory and handle emotional escalation
    emotion_memory_and_escalation(conversation_analysis)


# Run the project
if __name__ == "__main__":
    real_time_conversational_agent(duration=10)
"""
Example Inputs and Expected Outputs:
Example 1:
Input: Conversation with fluctuating emotions ("I can't believe this is happening, it's just so frustrating.")
Expected Output:

Transcription: "I can't believe this is happening, it's just so frustrating."
Conversation Analysis: "The speaker expresses frustration and anger."
Agent Response: The agent replies with a concerned tone using a slightly faster and louder voice, "I understand that you're feeling upset. Let's see how we can address this frustration."
Emotion Memory: If this frustration continues in further conversations, the agent escalates its response, perhaps suggesting solutions or comforting the speaker.
Example 2:
Input: Calm and positive conversation ("Things have been going well, I'm really happy with how everything is turning out.")
Expected Output:

Transcription: "Things have been going well, I'm really happy with how everything is turning out."
Conversation Analysis: "The speaker is expressing happiness and satisfaction."
Agent Response: The agent responds with an upbeat and faster voice, "That’s wonderful to hear! I'm so glad things are going well for you."
Key Improvements:
Multi-tiered emotional reactions: The agent not only detects emotions but responds dynamically based on specific emotional cues within the conversation.
Emotion memory and escalation: By tracking emotions over time, the agent adjusts its responses based on prior emotional states, simulating more natural human-like emotional progression.
Advanced text-to-speech integration: The project adjusts voice tone, speed, and volume based on the emotional content of the conversation, enhancing the realism of the interaction.
Real-time conversational flow analysis: The system analyzes not just static emotions, but how emotions change within a conversation, and generates responses that are contextually aware of these shifts.
"""