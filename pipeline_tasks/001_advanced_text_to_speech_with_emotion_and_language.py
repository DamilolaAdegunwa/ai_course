import os
from transformers import pipeline, logging
import certifi
import torchaudio  # Required for TTS and audio manipulation

# Enable detailed logging
logging.set_verbosity(logging.WARNING)

# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['ENV'] = 'prod'
os.environ['ENVIRONMENT'] = 'prod'

# Pipeline for emotion classification
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=0)

# Pipeline for language identification
language_classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=0)

# Pipeline for text-to-speech (TTS)
# tts_pipeline = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech", device=0)
tts_pipeline = pipeline("text-to-speech", device=0) # suno/bark-small


# Function to handle advanced TTS synthesis
def advanced_tts_synthesis(texts):
    results = []

    for text in texts:
        print(f"Processing: '{text}'")

        # Step 1: Detect emotion
        emotion = emotion_classifier(text)[0]['label']
        print(f"Detected Emotion: {emotion}")

        # Step 2: Detect language
        language = language_classifier(text)[0]['label']
        print(f"Detected Language: {language}")

        # Step 3: Generate speech based on detected emotion and language
        # Note: For simplicity, we assume English for now, but you can switch TTS models here based on the language detected.
        speech = tts_pipeline(text)

        # Store result with metadata
        results.append({
            "text": text,
            "emotion": emotion,
            "language": language,
            "speech": speech
        })

    return results


# Example usage of the function
texts_to_synthesize = [
    "I am so happy today! Let's celebrate!",
    "Estoy muy triste, no sé qué hacer...",
    "Get out of here! I am really angry with you.",
    "Je suis ravi de te rencontrer aujourd'hui.",
    "Life is beautiful when you appreciate the little things."
]

synthesis_results = advanced_tts_synthesis(texts_to_synthesize)

# Output the results (play or save the generated speech)
for result in synthesis_results:
    print(f"Text: {result['text']}")
    print(f"Emotion: {result['emotion']}")
    print(f"Language: {result['language']}")
    print("Speech generated successfully.\n")
    # You can use torchaudio or another library to save/play the speech.
