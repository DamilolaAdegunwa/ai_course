"""
Project Title: AI-Powered Real-Time Multilingual Speech Summarization and Contextual Response System
File Name: real_time_multilingual_speech_summarization.py

Project Description:
This project builds a real-time speech summarization system that:

Listens to multilingual audio input, transcribes it in real-time, and performs speech summarization.
Provides context-aware responses to user input based on both the content and sentiment of the speech.
Handles continuous audio streams, automatically segmenting and summarizing chunks of speech.
Incorporates real-time translation of the summarized speech into a target language (user-specified).
Responds with contextually relevant summaries or additional insights to continue the conversation intelligently.
Includes adaptive memory that summarizes longer conversations into key points for better context understanding.
This project adds complexity by handling long-form continuous audio, generating summarized speech output dynamically, and maintaining conversation context over time.

Python Code:

https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532
"""
import os
from openai import OpenAI
from apikey import apikey
import pyttsx3
from io import BytesIO
import pyaudio

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Memory storage for conversation context
conversation_memory = []

# Supported languages
supported_languages = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}

# Function to transcribe and summarize speech
def transcribe_and_summarize(audio_chunk, target_language="en"):
    audio_file = BytesIO(audio_chunk)

    # Step 1: Transcribe the speech
    transcription_response = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    transcription = transcription_response['text']
    detected_language = transcription_response.get('language', 'en')  # Default to English

    # Step 2: Translate the transcription if necessary
    if detected_language != target_language:
        translation_response = client.translations.create(
            model="whisper-1",
            source_language=detected_language,
            target_language=target_language,
            text=transcription
        )
        transcription = translation_response['translated_text']

    # Step 3: Summarize the speech
    summary_prompt = f"Summarize the following text: '{transcription}'. Provide a concise summary of the key points."
    summary_response = client.completions.create(
        model="text-davinci-003",
        prompt=summary_prompt,
        max_tokens=100
    )
    summary = summary_response['choices'][0]['text'].strip()

    return transcription, summary, detected_language

# Function to generate a context-aware response
def generate_contextual_response(user_summary):
    prompt = f"Based on the following summary: '{user_summary}', generate a response that continues the conversation in an intelligent and relevant way."
    response_generation = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )
    return response_generation['choices'][0]['text'].strip()

# Real-time audio recording
def record_audio_for_duration(duration=5, chunk_size=1024, channels=1, rate=16000):
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

# Function to maintain conversation memory
def update_conversation_memory(summary):
    conversation_memory.append(summary)
    if len(conversation_memory) > 10:  # Limit memory size to 10 summaries for now
        conversation_memory.pop(0)

# Function to provide a summary of the entire conversation
def summarize_conversation_memory():
    if not conversation_memory:
        return "No conversation data available."

    conversation_context = " ".join(conversation_memory)
    summary_response = client.completions.create(
        model="text-davinci-003",
        prompt=f"Summarize the following conversation: {conversation_context}",
        max_tokens=150
    )
    return summary_response['choices'][0]['text'].strip()

# Main chatbot function
def multilingual_speech_summarization_chatbot(target_language="en"):
    print("Welcome to the real-time speech summarization chatbot!")

    try:
        while True:
            # Step 1: Record the user's input
            audio_chunk = record_audio_for_duration(duration=5)

            # Step 2: Transcribe and summarize the user's input
            transcription, summary, detected_language = transcribe_and_summarize(audio_chunk, target_language=target_language)

            print(f"\nYou said (in {supported_languages.get(detected_language, 'unknown language')}): {transcription}")
            print(f"Summary: {summary}")

            # Update conversation memory with the latest summary
            update_conversation_memory(summary)

            # Step 3: Generate a context-aware response
            chatbot_response = generate_contextual_response(summary)
            print(f"Chatbot response: {chatbot_response}")

            # Step 4: Convert chatbot response to speech
            tts_engine.say(chatbot_response)
            tts_engine.runAndWait()

            # Step 5: Provide a summary of the conversation on request
            if "conversation summary" in transcription.lower():
                conversation_summary = summarize_conversation_memory()
                print(f"Conversation Summary: {conversation_summary}")
                tts_engine.say(conversation_summary)
                tts_engine.runAndWait()

            # Exit condition
            if "goodbye" in transcription.lower() or "bye" in transcription.lower():
                print("Goodbye! Ending the conversation.")
                tts_engine.say("Goodbye! Ending the conversation.")
                break

    except Exception as e:
        print(f"Error: {e}")

# Run the chatbot
if __name__ == "__main__":
    multilingual_speech_summarization_chatbot(target_language="en")
"""
Example Inputs and Expected Outputs:
Example 1:

Input (spoken in German): "Heute hatte ich ein sehr anstrengendes Meeting, aber es hat mir geholfen, einige neue Ideen zu entwickeln."
Expected Output:
Transcription (German): "Heute hatte ich ein sehr anstrengendes Meeting, aber es hat mir geholfen, einige neue Ideen zu entwickeln."
Translated to English: "Today I had a very exhausting meeting, but it helped me develop some new ideas."
Summary: "The user had a tiring meeting but gained new ideas."
Chatbot Response: "That sounds productive! What kind of new ideas did you come up with?"
Conversation Summary (if requested): "The user attended a tiring meeting but gained new ideas, and the conversation continued on that topic."
Example 2:

Input (spoken in Spanish): "La conferencia de hoy fue increíble. Aprendí mucho sobre inteligencia artificial y cómo aplicarla."
Expected Output:
Transcription (Spanish): "La conferencia de hoy fue increíble. Aprendí mucho sobre inteligencia artificial y cómo aplicarla."
Translated to English: "Today's conference was amazing. I learned a lot about artificial intelligence and how to apply it."
Summary: "The user learned a lot about artificial intelligence at a conference."
Chatbot Response: "That sounds fascinating! How do you plan to apply what you learned about AI?"
Conversation Summary (if requested): "The user attended a conference and learned about AI, with the conversation focusing on the practical applications of AI."
Key Features:
Real-Time Summarization: As the user speaks, the chatbot continuously transcribes, translates, and summarizes the speech.
Multilingual Support: The system can handle multiple languages, translating them in real-time to the target language.
Contextual Responses: The chatbot uses summarized speech to generate relevant responses that make the conversation more coherent and focused.
Conversation Memory: Tracks and summarizes the overall conversation, providing intelligent context retention over longer interactions.
Adaptive Conversations: The chatbot not only responds in real-time but adapts its responses to the context of previous discussions.
This project is noticeably more advanced than the previous ones by incorporating real-time summarization, contextual responses, and conversation memory, making it suitable for applications like AI meeting assistants, customer support chatbots, or personalized voice assistants.
"""