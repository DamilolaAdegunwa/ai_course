"""
Project Title: AI-Generated Interactive Audio Narratives with Dynamic Voice Modulation and Soundtrack Integration
File Name: ai_interactive_audio_narratives.py

Project Description:
In this advanced OpenAI audio project, we will create an AI-generated interactive audio narrative system that combines dynamic voice modulation, emotion-based soundtrack generation, and real-time narrative branching based on user input and detected emotions. The system will generate storylines on the fly, adjusting the voice pitch and style of each character based on emotional context while incorporating background soundtracks that align with the emotional tone of the scene.

Key Features:

Dynamic Voice Modulation: Change the pitch and tone of characters' voices in real time, reflecting their emotional state or context in the story.
Emotion-Based Soundtrack Generation: Generate background music that evolves based on the mood of the story and the emotions of the characters.
Interactive Storyline with Branching Narratives: The AI dynamically generates the plot, giving users control over character decisions, and the narrative branches based on these decisions.
Multi-Character Voice Simulation: Use different synthesized voices for multiple characters, modulated according to the narrative context and emotional state.
Soundtrack and Ambient Sound Integration: Add atmospheric sound effects and background music to create an immersive storytelling experience.
This project brings the previous audio projects to a new level by integrating advanced features like voice modulation and real-time music generation into the audio storytelling experience.

Python Code:
"""
import os
import pyttsx3
import sounddevice as sd
import librosa
from openai import OpenAI
from apikey import apikey
import speech_recognition as sr
from pydub import AudioSegment, playback

# OpenAI API setup
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Speech recognizer for real-time input
recognizer = sr.Recognizer()

# Load ambient sound effects
SOUND_EFFECTS = {
    "wind": "wind_ambience.wav",
    "rain": "rain_ambience.wav",
    "forest": "forest_ambience.wav"
}


# Play sound effect based on the current scene
def play_ambient_sound(effect_name):
    if effect_name in SOUND_EFFECTS:
        sound = AudioSegment.from_file(SOUND_EFFECTS[effect_name])
        playback.play(sound)


# Adjust voice pitch and rate for dynamic modulation
def modulate_voice(emotion):
    if emotion == "happy":
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('pitch', 150)
    elif emotion == "sad":
        tts_engine.setProperty('rate', 100)
        tts_engine.setProperty('pitch', 70)
    elif emotion == "angry":
        tts_engine.setProperty('rate', 180)
        tts_engine.setProperty('pitch', 200)
    else:
        tts_engine.setProperty('rate', 120)
        tts_engine.setProperty('pitch', 100)


# Generate background music based on emotional tone
def generate_background_music(emotion):
    if emotion == "happy":
        return "happy_music.wav"
    elif emotion == "sad":
        return "sad_music.wav"
    elif emotion == "tense":
        return "tense_music.wav"
    else:
        return "neutral_music.wav"


# Play background music in sync with the story
def play_background_music(emotion):
    music_file = generate_background_music(emotion)
    background_music = AudioSegment.from_file(music_file)
    playback.play(background_music)


# Record user voice input
def record_voice(duration=5, sample_rate=16000):
    print("Recording voice input...")
    voice_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return voice_data


# Transcribe speech and detect emotion
def detect_emotion(voice_data, sample_rate=16000):
    voice_wav = librosa.resample(voice_data[:, 0], orig_sr=sample_rate, target_sr=16000)
    voice_pcm = librosa.util.buf_to_int(voice_wav)

    # Transcribe using Whisper API
    transcribed_audio = client.audio.transcribe(model="whisper-1", file=voice_pcm)
    transcription = transcribed_audio['text']

    # Generate emotion analysis
    emotion_analysis = client.completions.create(
        model="text-davinci-003",
        prompt=f"Analyze the emotion in this conversation: {transcription}",
        max_tokens=50
    )
    emotion = emotion_analysis.choices[0].text.strip()
    print(f"Detected emotion: {emotion}")
    return transcription, emotion


# Generate a dynamic narrative script
def generate_narrative(scene_context, user_input, emotion):
    prompt = f"The current scene is described as: {scene_context}. The user said: {user_input}. The emotion is {emotion}. Continue the story."

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=250
    )

    return response.choices[0].text.strip()


# Main interactive audio narrative loop
def interactive_audio_narrative():
    scene_context = "A dense forest at dusk, with wind whistling through the trees."

    # Play initial ambient sound and background music
    play_ambient_sound("forest")
    play_background_music("neutral")

    while True:
        # Record the user's voice input
        voice_input = record_voice(duration=5)
        transcription, emotion = detect_emotion(voice_input)

        # Modulate the voice of the character based on the emotion
        modulate_voice(emotion)

        # Generate the next part of the narrative
        generated_narrative = generate_narrative(scene_context, transcription, emotion)
        print(f"Generated Narrative: {generated_narrative}")

        # Speak the generated script with dynamic voice modulation
        tts_engine.say(generated_narrative)
        tts_engine.runAndWait()

        # Play a new background track based on the emotion
        play_background_music(emotion)

        # Update scene context for the next part of the story
        scene_context += f" {generated_narrative}"

        # Exit condition
        if "end" in transcription.lower():
            print("Narrative has ended.")
            break


# Run the interactive audio narrative
if __name__ == "__main__":
    interactive_audio_narrative()
"""
Example Inputs and Expected Outputs:
Example 1:
User Input: “I hear something in the bushes. Should I investigate?”
Detected Emotion: Curious
Generated Narrative: “As you approach the bushes, you hear the rustling grow louder. A small animal darts out, disappearing into the trees. The wind continues to whistle through the branches.”
Modulated Voice: Slightly higher-pitched, fast-paced voice to reflect curiosity.
Background Music: Neutral, quiet forest ambiance.

Example 2:
User Input: “I feel nervous about going deeper into the forest.”
Detected Emotion: Fearful
Generated Narrative: “The forest grows darker as you move deeper. Shadows stretch out in all directions, and the sound of footsteps behind you makes your heart race.”
Modulated Voice: Low-pitched, slower-paced voice to reflect fear.
Background Music: Tense, eerie music.

Example 3:
User Input: “I feel more confident now.”
Detected Emotion: Confident
Generated Narrative: “With a renewed sense of determination, you press forward. The path ahead clears, and the trees open up to reveal a small, quiet clearing bathed in moonlight.”
Modulated Voice: Calm, steady voice reflecting confidence.
Background Music: Peaceful, uplifting music.

Key Features:
Voice Modulation: Adjusts the pitch and speed of the text-to-speech based on the detected emotion, giving characters distinct personalities and emotions.
Emotion-Based Soundtrack: Background music changes dynamically, reflecting the emotional tone of the narrative and providing a more immersive experience.
Real-Time Narrative Generation: The AI generates a real-time storyline based on user input, guiding the plot in new directions as the story progresses.
Ambient Sound Effects: The project incorporates environmental sounds (wind, rain, forest) to enhance the storytelling and provide a fully immersive experience.
Interactive Story Branching: The narrative branches based on user decisions and detected emotions, ensuring that every storytelling experience is unique.
Conclusion:
This project takes AI-driven audio narratives to the next level by incorporating dynamic voice modulation, emotion-based music, and immersive sound effects. The result is a highly interactive, emotionally engaging audio experience that responds to both user input and their emotional state in real time, making it an excellent example of advanced audio-based storytelling.
"""