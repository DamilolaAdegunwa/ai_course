"""
Project Title: Real-Time Audio Transcription and Dynamic Speaker Emotion-Driven Audio Manipulation
File Name: real_time_audio_transcription_emotion_audio_manipulation.py

Project Description:
This project builds upon the previous exercises by adding dynamic audio manipulation based on the emotions detected during real-time transcription. In addition to transcribing and detecting speaker sentiment, this project will:

Modify the audio playback in real-time based on the emotions of the speaker (e.g., adjusting pitch, speed, or adding audio effects depending on the detected sentiment).
Process audio in real-time as it is being recorded, giving immediate feedback not only in text and emotion, but also through transformed audio output.
Incorporate advanced voice modulation techniques to create more engaging and personalized audio experiences, where the emotional state of the speaker directly affects the way the audio is heard.
This project will use real-time streaming audio processing and includes components of advanced emotional understanding with dynamic modifications of the audio output.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from apikey import apikey

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to record audio
def record_audio(duration=10, sample_rate=16000):
    print("Recording audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording completed.")
    return recording


# Function to transcribe audio in real-time
def transcribe_audio(audio_data):
    audio_stream = BytesIO(audio_data)

    print("Transcribing audio...")
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )

    transcription = response['text']
    return transcription


# Function to detect emotion from transcription
def detect_emotion(transcription):
    print("Detecting emotion...")
    prompt = f"Analyze the emotion of this text: '{transcription}'. Provide one word: happy, sad, angry, or neutral."

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=5
    )

    emotion = response['choices'][0]['text'].strip().lower()
    return emotion


# Function to apply audio manipulation based on detected emotion
def manipulate_audio(emotion, audio_data):
    # Convert numpy array to audio
    wav.write("temp_audio.wav", 16000, audio_data.astype(np.int16))
    audio_segment = AudioSegment.from_wav("temp_audio.wav")

    # Manipulate audio based on emotion
    if emotion == "happy":
        print("Applying speed up for happy tone.")
        manipulated_audio = audio_segment.speedup(playback_speed=1.5)  # Speed up the audio
    elif emotion == "sad":
        print("Slowing down for sad tone.")
        manipulated_audio = audio_segment.speedup(playback_speed=0.8)  # Slow down the audio
    elif emotion == "angry":
        print("Lowering pitch for angry tone.")
        manipulated_audio = audio_segment.lower_pitch(5)  # Lower the pitch
    else:
        print("No manipulation for neutral tone.")
        manipulated_audio = audio_segment  # No changes for neutral emotion

    # Play manipulated audio
    play(manipulated_audio)


# Main function
def real_time_transcription_and_audio_modulation(duration=10):
    # Step 1: Record audio in real-time
    audio_data = record_audio(duration=duration)

    # Step 2: Transcribe audio to text
    audio_data_bytes = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
    transcription = transcribe_audio(audio_data_bytes.tobytes())
    print(f"Transcription: {transcription}")

    # Step 3: Detect emotion from transcription
    emotion = detect_emotion(transcription)
    print(f"Detected Emotion: {emotion}")

    # Step 4: Manipulate audio based on detected emotion
    manipulate_audio(emotion, audio_data)


# Run the project
if __name__ == "__main__":
    real_time_transcription_and_audio_modulation(duration=10)
"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A speaker saying:

"I'm so excited to be here! This is amazing!"
Expected Transcription:

css
Copy code
"I'm so excited to be here! This is amazing!"
Detected Emotion:

makefile
Copy code
Emotion: happy
Audio Manipulation:

The audio is sped up (1.5x playback speed) to match the upbeat tone of the speaker.
The modified audio plays back faster than the original.
Example 2:

Input Audio: A speaker saying:

"I can't believe this happened. It's so disappointing."
Expected Transcription:

arduino
Copy code
"I can't believe this happened. It's so disappointing."
Detected Emotion:

makefile
Copy code
Emotion: sad
Audio Manipulation:

The audio is slowed down (0.8x playback speed) to reflect the melancholic tone of the speaker.
The modified audio plays back more slowly than the original.
Example 3:

Input Audio: A speaker saying:

"This is so frustrating! I'm angry about how things are going!"
Expected Transcription:

arduino
Copy code
"This is so frustrating! I'm angry about how things are going!"
Detected Emotion:

makefile
Copy code
Emotion: angry
Audio Manipulation:

The pitch of the audio is lowered to reflect the intensity and anger in the speaker's voice.
The modified audio has a lower pitch than the original recording.
Key Features:
Real-Time Audio Transcription: Capture audio in real-time, immediately convert it to text using OpenAI's Whisper API.
Emotion Detection from Speech: Detect and classify the speaker's emotional state based on the transcribed text.
Dynamic Audio Manipulation: Based on the detected emotion, apply modifications to the recorded audio, such as changing pitch, speed, or adding effects like reverb.
Advanced Audio Processing: Using libraries like Pydub, the project manipulates audio based on emotions and provides feedback by playing the modified audio.
Real-Time Interactive Feedback: The system not only transcribes and translates audio but also modifies the playback to align with the sentiment conveyed in speech.
Use Cases:
Interactive Storytelling: This system could be used to create more immersive experiences in storytelling, where characters' emotions dynamically affect how their voices sound.
Voice-Driven Games: In gaming, the emotional state of characters could affect the voice and tone dynamically, increasing realism.
Customer Feedback in Call Centers: For customer service applications, the tool can detect and react to customer emotions, modulating the voice to reflect empathy or assertiveness.
Theatrical Performances: In live performances or theater, this system could add a layer of emotion-driven audio manipulation to enhance the audience’s experience.
This project introduces dynamic audio manipulation based on detected emotions, adding a real-time audio feedback loop where speaker emotions drive the transformation of their voice. This level of complexity brings a new dimension to interactive, real-time audio processing.
"""