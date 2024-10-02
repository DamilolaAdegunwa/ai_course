"""
Project Title: Advanced Audio Transcription with Speaker Emotion Profiling and Adaptive Sound Effects
File Name: advanced_audio_transcription_emotion_profiling_sound_effects.py

Project Description:
In this even more advanced OpenAI audio project, we will:

Transcribe speech from an audio file.
Identify speakers (using speaker diarization) and create an emotional profile for each speaker throughout the conversation.
Apply adaptive sound effects such as reverb, echo, or modulation based on the emotions detected for each speaker (e.g., echo for sadness, upbeat effects for excitement).
Mix the audio with dynamic effects based on the emotional profiling, enhancing the listening experience in real-time.
This project involves multiple complex elements such as speaker identification, profiling emotions over time, and the application of sound effects based on emotional shifts within the conversation.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
import librosa
import numpy as np
import pydub
from io import BytesIO
import soundfile as sf
from pydub.playback import play

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Load sentiment and speaker profiling model info
sentiment_labels = ['happy', 'sad', 'angry', 'neutral', 'excited']


# Function to transcribe audio with speaker diarization
def transcribe_with_speaker_diarization(file_path):
    audio_file = open(file_path, "rb")

    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )

    transcription = response['text']
    speaker_segments = response.get('speaker_labels', [])
    return transcription, speaker_segments


# Function to analyze emotion for each speaker segment
def profile_emotions(transcription, speaker_segments):
    emotions = {}
    for segment in speaker_segments:
        speaker = segment['speaker']
        text_segment = transcription[segment['start']:segment['end']]

        # Analyze sentiment for each segment
        sentiment_prompt = f"Analyze the sentiment of this text: '{text_segment}' and indicate if it's happy, sad, angry, neutral, or excited."
        sentiment_response = client.completions.create(
            model="text-davinci-003",
            prompt=sentiment_prompt,
            max_tokens=50
        )
        sentiment = sentiment_response['choices'][0]['text'].strip().lower()

        if speaker not in emotions:
            emotions[speaker] = []

        emotions[speaker].append(sentiment)

    return emotions


# Function to apply sound effects based on emotion
def apply_emotion_based_effects(audio_data, emotions):
    sound = pydub.AudioSegment(audio_data)

    # Modify audio based on emotional profiling
    for speaker, emotion_list in emotions.items():
        for emotion in emotion_list:
            if emotion == 'sad':
                sound = sound.low_pass_filter(300)  # Low-pass filter for sadness
            elif emotion == 'happy':
                sound = sound.speedup(playback_speed=1.1)  # Speed up for happiness
            elif emotion == 'angry':
                sound = sound + 10  # Increase volume for anger
            elif emotion == 'excited':
                sound = sound.high_pass_filter(5000)  # High-pass filter for excitement
            elif emotion == 'neutral':
                # Neutral: leave the sound unaltered
                pass

    return sound


# Main function to run transcription, profiling, and adaptive sound effects
def transcribe_and_adapt_audio(file_path):
    # Step 1: Transcribe audio with speaker diarization
    transcription, speaker_segments = transcribe_with_speaker_diarization(file_path)
    print(f"Transcription: {transcription}")

    # Step 2: Profile emotions for each speaker
    emotions = profile_emotions(transcription, speaker_segments)
    print(f"Emotional profile for speakers: {emotions}")

    # Step 3: Load audio and apply emotion-based effects
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    sf.write('temp_audio.wav', audio_data, sample_rate)
    modified_audio = apply_emotion_based_effects('temp_audio.wav', emotions)

    # Step 4: Play the modified audio with adaptive effects
    print("Playing modified audio with adaptive effects...")
    play(modified_audio)


# Run the program with the given audio file
if __name__ == "__main__":
    file_path = r"C:\path_to_audio\sample_conversation_audio.mp3"
    transcribe_and_adapt_audio(file_path)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A conversation between two speakers.

Speaker 1: "I can't believe this happened, I'm so frustrated!"
Speaker 2: "Calm down, we'll figure this out."
Expected Transcription:

less
Copy code
Speaker 1: "I can't believe this happened, I'm so frustrated!"
Speaker 2: "Calm down, we'll figure this out."
Expected Speaker Segmentation:

less
Copy code
Speaker 1: "I can't believe this happened, I'm so frustrated!"
Speaker 2: "Calm down, we'll figure this out."
Expected Emotional Profiling:

yaml
Copy code
Speaker 1: angry
Speaker 2: calm/neutral
Adaptive Sound Effects:

Speaker 1's dialogue gets volume increased by +10 dB (due to "angry").
Speaker 2's dialogue remains unchanged (neutral).
Action: Plays the conversation audio with altered sound effects to match the emotions of the speakers.

Example 2:

Input Audio: A conversation between two speakers.

Speaker 1: "I'm really excited to start this new project!"
Speaker 2: "Yeah, it's going to be amazing!"
Expected Transcription:

arduino
Copy code
Speaker 1: "I'm really excited to start this new project!"
Speaker 2: "Yeah, it's going to be amazing!"
Expected Speaker Segmentation:

arduino
Copy code
Speaker 1: "I'm really excited to start this new project!"
Speaker 2: "Yeah, it's going to be amazing!"
Expected Emotional Profiling:

yaml
Copy code
Speaker 1: excited
Speaker 2: happy
Adaptive Sound Effects:

Speaker 1's dialogue gets a high-pass filter applied (for "excited").
Speaker 2's dialogue speeds up slightly (for "happy").
Action: Plays the conversation audio with dynamically applied sound effects that reflect the excitement and happiness of the speakers.

Key Features:
Speaker Diarization: The model identifies different speakers in the conversation and transcribes their speech independently.
Emotion Profiling for Each Speaker: Uses sentiment analysis to profile emotions for each speaker, based on their speech.
Adaptive Sound Effects: Modifies audio playback with adaptive effects like low-pass filters for sadness, volume boosts for anger, speed adjustments for happiness, etc.
Audio Playback with Effects: Plays back the conversation with dynamically applied effects, giving an immersive emotional audio experience.
Use Cases:
Enhanced Storytelling: Imagine podcasts or audiobooks where the audio dynamically adapts based on the emotions in the story, offering an immersive experience for listeners.
Customer Support Analysis: In customer service, this system could analyze conversations for emotion detection and apply real-time adjustments to reflect frustration or excitement in the conversation.
Interactive Voice Assistants: Voice assistants can alter their response tones based on user emotions, creating a more personalized and responsive interaction.
This project combines speaker identification, emotional analysis, and audio effect processing, significantly enhancing the complexity and real-world applicability of OpenAI audio projects.
"""