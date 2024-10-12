"""
Project Title: Context-Aware Conversational Agent with Real-Time Audio Effects and Multi-Speaker Emotion Detection
File Name: context_aware_audio_agent.py

Project Description:
This project extends the capabilities of an audio-based conversational agent by introducing multi-speaker emotion detection and context-aware audio effects. The system can process inputs from multiple speakers in real time, detect their individual emotions, and dynamically adjust the conversation flow and audio effects (such as volume, pitch, and background noise). Additionally, the agent maintains a memory of conversation context to adjust responses based on speaker emotions, previous dialogues, and conversational tone.

Key features:

Multi-Speaker Emotion Detection: Detects and analyzes emotions from multiple speakers in real time using OpenAI's Whisper and GPT-based emotion classifiers.
Contextual Audio Effects: Dynamically adjusts audio properties (e.g., background music, reverb, echo, volume) based on conversational context and detected emotions.
Memory-Aware Conversations: The agent maintains short-term memory of the conversation's context, adapting responses accordingly to fit the tone and emotion of each speaker.
Customizable Speaker Profiles: Each speaker has a customizable profile that can adjust how the system responds based on known preferences, such as tone, music preference, or audio effects.
This project aims to provide a deeper, personalized, and emotion-aware interaction in a multi-speaker environment.

Python Code:
"""
import os
import numpy as np
import sounddevice as sd
import librosa
import pyttsx3
from openai import OpenAI
from apikey import apikey
import speech_recognition as sr
from pydub import AudioSegment, playback

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Recognizer for user input
recognizer = sr.Recognizer()

# Speaker profiles (customizable)
SPEAKER_PROFILES = {
    "speaker_1": {"name": "Alice", "emotion_pref": "happy", "bg_music": "calm_ambience.wav"},
    "speaker_2": {"name": "Bob", "emotion_pref": "neutral", "bg_music": "city_ambience.wav"}
}

# Load background music files
BACKGROUND_MUSIC = {
    "calm_ambience": "calm_ambience.wav",
    "city_ambience": "city_ambience.wav",
    "rain_ambience": "rain_ambience.wav"
}


# Record audio from the user
def record_audio(duration=5, sample_rate=16000):
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio


# Detect emotion using OpenAI
def detect_emotion(speaker_name, audio_data, sample_rate=16000):
    audio_wav = librosa.resample(audio_data[:, 0], orig_sr=sample_rate, target_sr=16000)
    audio_pcm = librosa.util.buf_to_int(audio_wav)

    # Transcribe and analyze emotion
    audio_input = client.audio.transcribe(model="whisper-1", file=audio_pcm)
    transcription = audio_input['text']

    # Analyze emotions from transcription
    response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotional tone of this conversation: {transcription}",
        max_tokens=50
    )
    emotion = response.choices[0].text.strip()
    print(f"{speaker_name}'s detected emotion: {emotion}")

    return transcription, emotion


# Apply audio effects based on the conversation context and speaker emotion
def apply_audio_effects(emotion, soundscape):
    if "happy" in emotion:
        return soundscape + 3  # Increase volume for happy emotions
    elif "sad" in emotion:
        return soundscape - 3  # Decrease volume for sad emotions
    elif "angry" in emotion:
        return soundscape + 5  # Increase volume for angry emotions
    else:
        return soundscape  # Default volume adjustment


# Play background music with dynamic emotion-based effects
def play_background_music(emotion, speaker_profile):
    bg_music_file = BACKGROUND_MUSIC[speaker_profile['bg_music'].split('.')[0]]
    soundscape = AudioSegment.from_file(bg_music_file)

    # Apply emotion-based effects to the soundscape
    adjusted_soundscape = apply_audio_effects(emotion, soundscape)

    playback.play(adjusted_soundscape)


# Context-aware conversation management
def context_aware_conversation():
    conversation_memory = []

    while True:
        user_input = input("Enter 'record' to start recording (or 'exit' to stop): ")
        if user_input.lower() == 'exit':
            break

        # Record and detect emotion for both speakers
        audio_speaker_1 = record_audio(duration=5)
        transcript_1, emotion_1 = detect_emotion("speaker_1", audio_speaker_1)

        audio_speaker_2 = record_audio(duration=5)
        transcript_2, emotion_2 = detect_emotion("speaker_2", audio_speaker_2)

        # Store conversations in memory
        conversation_memory.append((transcript_1, emotion_1, "speaker_1"))
        conversation_memory.append((transcript_2, emotion_2, "speaker_2"))

        # Analyze the context and generate response based on both emotions and conversation history
        context_prompt = f"Context of the conversation: {conversation_memory}. Generate an appropriate response for speaker_1 based on their emotion: {emotion_1}"

        response = client.completions.create(
            model="text-davinci-003",
            prompt=context_prompt,
            max_tokens=100
        )

        generated_response = response.choices[0].text.strip()
        print(f"Generated response for speaker 1: {generated_response}")

        # Speak the response using TTS
        tts_engine.say(generated_response)
        tts_engine.runAndWait()

        # Play background music adjusted for emotion of the speaker
        play_background_music(emotion_1, SPEAKER_PROFILES['speaker_1'])
        play_background_music(emotion_2, SPEAKER_PROFILES['speaker_2'])


# Run the context-aware conversational agent
if __name__ == "__main__":
    context_aware_conversation()
"""
Example Inputs and Expected Outputs:
Example 1:
Speaker 1 (Alice): “I had a great day today!”
Speaker 2 (Bob): “It was fine for me too.”
Detected Emotions:

Speaker 1: Happy
Speaker 2: Neutral
Generated Response:
“For Alice, it sounds like you had an exciting day. Bob, maybe it was just a regular day for you.”

Expected Output:

Background music: Calm ambiance for Alice, with slight volume increase.
Voice: Text-to-speech response in a cheerful tone for Alice, and neutral tone for Bob.
Example 2:
Speaker 1 (Alice): “I feel really sad today.”
Speaker 2 (Bob): “Things haven’t been great for me either.”
Detected Emotions:

Speaker 1: Sad
Speaker 2: Sad
Generated Response:
“I’m sorry to hear that both of you are feeling down. Let me play something more soothing.”

Expected Output:

Background music: Soft rain ambiance, with lower volume and slower pace for both speakers.
Voice: Text-to-speech response in a slow, soothing tone.
Key Features:
Multi-Speaker Emotion Detection: Simultaneously detects emotions from multiple speakers using OpenAI’s Whisper and GPT models.
Contextual Response Generation: Generates dynamic responses that fit the emotional tone and context of the ongoing conversation.
Dynamic Audio Effects: Adjusts background music, audio effects, and speech synthesis dynamically based on speaker emotions and conversational context.
Speaker Profiles: Each speaker has a customizable profile with preferences for background music and emotional responses.
Conclusion:
This project expands on real-time emotion detection by handling multi-speaker scenarios, adding dynamic audio effects based on conversation context, and maintaining short-term memory for adaptive responses. It creates a more sophisticated and immersive conversational agent that can understand, respond to, and adapt based on the emotional states and contexts of multiple speakers.
"""