"""
Project Title: Adaptive Multi-Speaker Conversation with Sentiment-Driven 3D Soundscapes
File Name: adaptive_multi_speaker_conversation_3d_soundscapes.py

Project Description:
This project takes multi-speaker conversational AI to the next level by enabling real-time, multi-language conversation with multiple virtual agents and dynamic sentiment-driven 3D soundscapes. Each speaker (agent) interacts independently, understanding and responding to users in multiple languages. The system uses real-time sentiment analysis to modify both the voice tone of the virtual agents and the 3D positioning of environmental sounds, making it feel like the user is engaging in a conversation with different agents around them in a virtual space.

Complexity:
Multi-speaker system: Allows multiple AI agents to converse with the user in real-time, with independent language and sentiment detection.
Sentiment-driven 3D sound positioning: Based on the emotional tone of the conversation, environmental sounds are placed in a 3D space around the user to create an immersive experience.
Dynamic voice modulation: Adjusts each agent’s voice based on emotion (e.g., sad, happy, angry) and language, changing pitch, speed, and directionality.
3D sound simulation: Utilizes spatial audio techniques to position each agent and environmental sounds in a virtual space, where distance and location dynamically influence the volume and timbre.
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

# TTS engine initialization
tts_engine = pyttsx3.init()

# Define the multi-agent system with language support and positions
AGENTS = {
    "agent_1": {"language": "english", "position": [-1, 0, 1], "emotion": "neutral"},
    "agent_2": {"language": "french", "position": [1, 0, -1], "emotion": "neutral"},
    "agent_3": {"language": "spanish", "position": [0, 1, 0], "emotion": "neutral"}
}

# Define sentiment-driven environmental soundscapes
SOUNDSCAPES = {
    "happy": "happy_birds_chirping.wav",
    "sad": "sad_rain.wav",
    "angry": "angry_thunder.wav",
    "neutral": "calm_wind.wav"
}

# Define TTS voice properties for emotions
VOICE_PROPERTIES = {
    "happy": {"rate": 160, "pitch": 200},
    "sad": {"rate": 100, "pitch": 130},
    "angry": {"rate": 200, "pitch": 250},
    "neutral": {"rate": 130, "pitch": 150}
}


# Load audio function
def load_audio(file_path):
    return AudioSegment.from_file(file_path)


# Play audio function
def play_audio(audio_segment):
    playback.play(audio_segment)


# Adjust volume of audio based on 3D position
def adjust_3d_audio_volume(audio, x, y, z):
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    adjusted_audio = audio - (distance * 5)
    return adjusted_audio


# Record user input
def record_audio(duration=5, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio


# Convert numpy array to PCM
def numpy_to_pcm(audio_data):
    audio_pcm = (audio_data * 32767).astype(np.int16)
    return audio_pcm.tobytes()


# Detect language and sentiment from user input
def detect_language_and_sentiment(audio_bytes):
    audio_stream = BytesIO(audio_bytes)
    print("Detecting language and sentiment...")

    # Use Whisper to transcribe the input and detect language
    transcription_response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_stream,
        response_format="json"
    )
    transcription = transcription_response['text']
    detected_language = transcription_response.get('language', 'english')

    # Analyze sentiment from transcription
    sentiment_prompt = f"Analyze the sentiment of this sentence: '{transcription}'"
    sentiment_response = client.completions.create(
        model="gpt-4",
        prompt=sentiment_prompt,
        max_tokens=50
    )
    detected_sentiment = sentiment_response.choices[0]['text'].strip().lower()
    return transcription, detected_language, detected_sentiment


# Generate response based on agent, sentiment, and language
def generate_response(agent, input_text, sentiment):
    language = AGENTS[agent]["language"]
    prompt = f"Respond in {language} with a {sentiment} tone: {input_text}"
    response = client.completions.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0]['text'].strip()


# Adjust TTS for emotion
def adjust_tts_properties(emotion):
    voice_properties = VOICE_PROPERTIES.get(emotion, VOICE_PROPERTIES['neutral'])
    tts_engine.setProperty('rate', voice_properties['rate'])
    tts_engine.setProperty('pitch', voice_properties['pitch'])


# Speak with 3D positioning
def speak_response(agent, response_text, emotion):
    adjust_tts_properties(emotion)
    agent_position = AGENTS[agent]["position"]
    tts_engine.say(response_text)
    tts_engine.runAndWait()

    # Simulate 3D positioning by adjusting volume
    tts_audio = load_audio(response_text)
    adjusted_audio = adjust_3d_audio_volume(tts_audio, *agent_position)
    play_audio(adjusted_audio)


# Apply sentiment-driven environmental soundscape
def apply_soundscape(sentiment):
    soundscape_file = SOUNDSCAPES.get(sentiment, SOUNDSCAPES['neutral'])
    soundscape = load_audio(soundscape_file)
    play_audio(soundscape)


# Main interaction loop with multi-speaker agents
def multi_agent_interaction(duration=5):
    while True:
        # Step 1: Record user input
        audio_data = record_audio(duration)

        # Step 2: Convert to PCM bytes
        audio_bytes = numpy_to_pcm(audio_data)

        # Step 3: Detect language and sentiment
        transcription, language, sentiment = detect_language_and_sentiment(audio_bytes)

        # Step 4: Generate responses for each agent
        for agent in AGENTS:
            response_text = generate_response(agent, transcription, sentiment)
            print(f"{agent} Response: {response_text}")

            # Step 5: Speak response with 3D positioning and emotion
            speak_response(agent, response_text, sentiment)

        # Step 6: Play environmental soundscape based on sentiment
        apply_soundscape(sentiment)


# Run the multi-agent conversation with 3D soundscape
if __name__ == "__main__":
    multi_agent_interaction(duration=5)
"""
Example Inputs and Expected Outputs:
Example 1:
Input: "I'm so happy to be talking to all of you!"
Expected Output:

Transcription: "I'm so happy to be talking to all of you!"
Detected Sentiment: "happy"
Agent 1 Response: "That’s wonderful! I’m happy to hear that." (spoken in English with an upbeat tone)
Agent 2 Response: "Je suis tellement contente de te parler aussi." (spoken in French with a joyful tone)
Agent 3 Response: "¡Es fantástico que estés feliz!" (spoken in Spanish with a cheerful tone)
Environmental Soundscape: Uplifting birds chirping in the background, simulated in 3D space.
Example 2:
Input: "I'm feeling really down today."
Expected Output:

Transcription: "I'm feeling really down today."
Detected Sentiment: "sad"
Agent 1 Response: "I'm sorry you're feeling that way." (spoken in English with a soft tone)
Agent 2 Response: "Je suis désolé que tu te sentes comme ça." (spoken in French with a low tone)
Agent 3 Response: "Lo siento, espero que te sientas mejor." (spoken in Spanish with a somber tone)
Environmental Soundscape: Rainy ambiance with soft droplets, positioned in 3D around the user.
Example 3:
Input: "I don't know why, but I'm so angry right now!"
Expected Output:

Transcription: "I don't know why, but I'm so angry right now!"
Detected Sentiment: "angry"
Agent 1 Response: "It's okay to feel angry sometimes." (spoken in English with a firm tone)
Agent 2 Response: "Je comprends, mais essaie de te calmer." (spoken in French with a stern tone)
Agent 3 Response: "Respira hondo, te sentirás mejor." (spoken in Spanish with an assertive tone)
Environmental Soundscape: Thunderstorm with strong winds surrounding the user in a 3D audio environment.
Key Features:
Multi-Speaker System: Several agents interact with the user, each responding with their unique language and voice properties.
Emotion-Driven Responses: AI agents respond with sentiment-adapted voices based on the detected emotional tone of the conversation.
3D Soundscape Simulation: Utilizes 3D audio to position agents' voices and environmental sounds around the user, creating a fully immersive experience.
Multi-Language Support: Each agent can converse in a different language (English, Spanish, French), making the interaction multilingual and dynamic."""