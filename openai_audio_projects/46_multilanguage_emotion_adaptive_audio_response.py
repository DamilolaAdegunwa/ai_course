"""
Project Title: Multi-Language Conversational AI with Emotionally Adaptive Audio Response
File Name: multilanguage_emotion_adaptive_audio_response.py

Project Description:
This project takes the real-time audio interaction to a whole new level by integrating multi-language conversational capabilities with emotionally adaptive responses. It allows a user to interact in different languages, and the system detects both language and emotional tone to generate contextually appropriate responses with dynamic emotional adjustments in voice tone and background soundscapes. The system reacts not only by choosing suitable language but also by modifying the emotion of the responses based on the detected emotion of the user.

This exercise is advanced due to:

Multi-language capability: It detects and switches between multiple languages (e.g., English, Spanish, French) in real-time, offering a seamless interaction experience.
Emotion-to-voice modulation: The voice response dynamically changes tone, pacing, and timbre based on emotional analysis, simulating real human emotion.
Real-time audio synthesis: The system uses a combination of emotion detection, voice generation, and multi-language support to synthesize realistic, emotion-adaptive audio.
Emotion-driven environmental soundscape: The response is layered with emotionally reactive background music or soundscapes to enhance immersion, making the interaction more lifelike and emotionally engaging.
Python Code:
"""
import os
import numpy as np
from openai import OpenAI
from apikey import apikey
from io import BytesIO
from pydub import AudioSegment, playback
import pyttsx3
import sounddevice as sd

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize TTS for multi-language response
tts_engine = pyttsx3.init()

# Emotion-based soundscapes
SOUNDSCAPES = {
    "happy": "happy_background_music.wav",
    "sad": "sad_ambiance.wav",
    "angry": "angry_wind.wav",
    "neutral": "neutral_ambiance.wav"
}

# Initialize language support (e.g., English, Spanish, French)
SUPPORTED_LANGUAGES = {
    "english": "en",
    "spanish": "es",
    "french": "fr"
}

# Set voice properties for different languages and emotional states
VOICE_PROPERTIES = {
    "happy": {"rate": 150, "pitch": 200},
    "sad": {"rate": 100, "pitch": 120},
    "angry": {"rate": 200, "pitch": 250},
    "neutral": {"rate": 130, "pitch": 150}
}


# Function to load audio file
def load_audio(file_path):
    return AudioSegment.from_file(file_path)


# Function to play an audio file
def play_audio(audio_segment):
    playback.play(audio_segment)


# Function to apply emotion-based audio adjustments
def apply_emotion_audio(emotion):
    soundscape_file = SOUNDSCAPES.get(emotion, SOUNDSCAPES['neutral'])
    soundscape = load_audio(soundscape_file)
    play_audio(soundscape)


# Record user input
def record_audio(duration=10, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio


# Convert numpy array to PCM bytes
def numpy_to_pcm(audio_data):
    audio_pcm = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
    return audio_pcm.tobytes()


# Detect language and emotion from user input
def detect_language_and_emotion(audio_bytes):
    audio_stream = BytesIO(audio_bytes)
    print("Detecting language and emotion...")

    # Use Whisper to detect transcription and language
    transcription_response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )

    transcription = transcription_response['text']
    detected_language = transcription_response.get('language', 'english')

    # Analyze emotion from the transcription
    emotion_prompt = f"Analyze the emotional tone of the following text: '{transcription}'"
    emotion_response = client.completions.create(
        model="gpt-4",
        prompt=emotion_prompt,
        max_tokens=50
    )

    detected_emotion = emotion_response.choices[0]['text'].strip().lower()
    print(f"Detected Language: {detected_language}, Detected Emotion: {detected_emotion}")
    return transcription, detected_language, detected_emotion


# Generate emotionally adapted text based on user input
def generate_emotion_adapted_text(language, emotion, input_prompt):
    prompt = f"Respond in {language} to the following user input with a {emotion} tone: {input_prompt}"
    response = client.completions.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0]['text'].strip()


# Adjust TTS properties based on emotion
def adjust_tts_properties(emotion):
    voice_properties = VOICE_PROPERTIES.get(emotion, VOICE_PROPERTIES['neutral'])
    tts_engine.setProperty('rate', voice_properties['rate'])
    tts_engine.setProperty('pitch', voice_properties['pitch'])


# Speak the response with emotion-adaptive voice
def speak_response(response_text, language, emotion):
    adjust_tts_properties(emotion)
    tts_engine.say(response_text, lang=SUPPORTED_LANGUAGES[language])
    tts_engine.runAndWait()


# Main interaction loop
def real_time_multilanguage_interaction(duration=10):
    while True:
        # Step 1: Record user input
        audio_data = record_audio(duration)

        # Step 2: Convert numpy array to PCM bytes for Whisper input
        audio_bytes = numpy_to_pcm(audio_data)

        # Step 3: Detect language and emotion from user input
        transcription, language, emotion = detect_language_and_emotion(audio_bytes)

        # Step 4: Generate emotionally adaptive response
        input_prompt = transcription
        response_text = generate_emotion_adapted_text(language, emotion, input_prompt)
        print(f"Generated Response: {response_text}")

        # Step 5: Speak the response in the detected language and emotion
        speak_response(response_text, language, emotion)

        # Step 6: Play an emotion-based background soundscape
        apply_emotion_audio(emotion)


# Run the multi-language, emotion-driven audio system
if __name__ == "__main__":
    real_time_multilanguage_interaction(duration=10)
"""
Example Inputs and Expected Outputs:
Example 1:
Input (spoken in Spanish): "Estoy un poco triste hoy, me siento solo."
Expected Output:

Transcription: "Estoy un poco triste hoy, me siento solo."
Detected Language: "spanish"
Detected Emotion: "sad"
Generated Response: "Lo siento mucho. Estoy aquí para ti."
Voice Properties: TTS will speak slowly with a lower pitch to match the "sad" tone.
Background Soundscape: Plays a slow, sad ambiance with soft rain sounds.
Example 2:
Input (spoken in English): "I'm so happy! Today is a great day!"
Expected Output:

Transcription: "I'm so happy! Today is a great day!"
Detected Language: "english"
Detected Emotion: "happy"
Generated Response: "That sounds amazing! I'm so happy for you!"
Voice Properties: TTS will speak with an upbeat tone, faster rate, and higher pitch.
Background Soundscape: Plays uplifting, joyful background music.
Example 3:
Input (spoken in French): "Je suis en colère, je n'aime pas ça!"
Expected Output:

Transcription: "Je suis en colère, je n'aime pas ça!"
Detected Language: "french"
Detected Emotion: "angry"
Generated Response: "Je comprends que tu sois en colère, je suis là pour t'aider."
Voice Properties: TTS will speak with a fast, sharp tone and increased pitch to reflect anger.
Background Soundscape: Plays an intense, storm-like soundscape with wind and thunder.
Key Features:
Multi-Language Support: Allows the user to speak in multiple languages (English, Spanish, French), and the system detects and responds accordingly.
Emotion-Adaptive TTS: Voice response dynamically adapts to the emotional context, adjusting speed, pitch, and timbre for realistic interactions.
Emotion-Driven Background Sound: Adds emotionally reactive background soundscapes to enhance immersion based on the user's emotional state.
Real-Time Interaction: Continuous, real-time audio interaction loop with on-the-fly emotion and language detection.
"""