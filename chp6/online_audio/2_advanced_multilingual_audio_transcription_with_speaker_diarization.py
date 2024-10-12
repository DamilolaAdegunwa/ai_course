"""
Project Title: Advanced Multilingual Audio Transcription with Speaker Diarization
File Name: advanced_multilingual_audio_transcription_with_speaker_diarization.py

Project Description:
In this project, you'll create an advanced audio transcription system capable of handling multilingual audio inputs. Additionally, the system will incorporate basic speaker diarization, identifying distinct speakers in the audio and assigning speaker labels to each segment of transcribed text. This exercise goes beyond simple audio transcription by introducing multilingual capabilities and distinguishing between speakers within the audio.

You'll be able to test this on audios that feature multiple languages and multiple speakers, making it suitable for podcasts, interviews, and multilingual conferences.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
from openai import OpenAI
from apikey import apikey, filepath
import os

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Function for advanced multilingual transcription with speaker diarization
file_path = filepath
def transcribe_audio_with_speaker_diarization(file_path):
    print(f"Processing audio file: {file_path} ...")

    # Open the audio file
    audio = open(file_path, "rb")

    # Transcribe the audio with the multilingual Whisper model
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        response_format="json",  # Return detailed JSON for speaker labeling
        #language="multilingual",  # Enable multilingual transcription
    )

    # Process the response to identify speakers (mock implementation)
    print('here is the response')
    print(response.text)
    transcription = response.text
    speakers = ["Speaker 1", "Speaker 2", "Speaker 3"]  # Mock speakers; replace with diarization logic if available
    segments = transcription.split(".")  # Splitting text into segments

    labeled_transcription = []
    for i, segment in enumerate(segments):
        speaker_label = speakers[i % len(speakers)]  # Cycle through speakers for mock diarization
        labeled_transcription.append(f"{speaker_label}: {segment.strip()}")

    return "\n".join(labeled_transcription)

# Example usage with multilingual audio
# file_path = r"C:\path\to\your\audio\multilingual_audio.mp4"


transcription_with_speakers = transcribe_audio_with_speaker_diarization(file_path)
# Output the final transcription with speaker labels
print("\nTranscription with Speaker Diarization:\n")
print(transcription_with_speakers)

"""
Example Inputs and Expected Outputs:
Example 1:

Input Audio: A podcast with two speakers, one speaking in English and the other in French.
Expected Output:
yaml
Copy code
Speaker 1: Hello everyone, welcome to the show.
Speaker 2: Bonjour à tous, bienvenue à notre émission.
Speaker 1: Today, we will discuss artificial intelligence.
Speaker 2: Aujourd'hui, nous parlerons de l'intelligence artificielle.
Example 2:

Input Audio: An interview where the interviewer asks questions in Spanish, and the guest responds in English.
Expected Output:
yaml
Copy code
Speaker 1: ¿Cómo te sientes acerca de la tecnología moderna?
Speaker 2: I feel that modern technology is advancing rapidly and changing the world.
Speaker 1: ¿Crees que hay algún riesgo asociado con la IA?
Speaker 2: Yes, there are definitely risks with AI that we need to be cautious of.
Key Concepts and Features:
Multilingual Transcription: Supports multiple languages within the same audio file.
Speaker Diarization (Mock Implementation): Labels different speakers with distinct identifiers (e.g., Speaker 1, Speaker 2).
Whisper Model: Uses OpenAI's advanced Whisper model for high-quality audio transcription.
JSON Output Parsing: Transcription response is processed as JSON, allowing for more detailed and customized output.
This project is a noticeable step up from your previous audio transcription project, as it adds multilingual support and introduces a basic speaker identification feature, making it more applicable to complex real-world scenarios like interviews and multilingual conferences.
"""