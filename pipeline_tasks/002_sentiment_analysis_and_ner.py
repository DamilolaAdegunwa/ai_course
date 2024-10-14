import os
from transformers import pipeline, logging
import certifi

# Enable detailed logging
# logging.set_verbosity_info() # logging.DEBUG, logging.INFO
logging.set_verbosity(logging.WARNING)


# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['ENV'] = 'dev'
os.environ['ENVIRONMENT'] = 'dev'

# Define pipelines
sentiment_classifier = pipeline(task="sentiment-analysis", device=0, model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
ner_classifier = pipeline(task="ner", grouped_entities=True, device=0, model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Example text to analyze
texts = [
    "Elon Musk founded SpaceX and Tesla.",
    "The Eiffel Tower is in Paris, and it’s a beautiful landmark.",
    "I love programming in Python! It's amazing.",
    "The stock market crashed yesterday, and it was devastating.",
    "Barack Obama was the 44th President of the United States."
]

# Sentiment Analysis
print("Sentiment Analysis Results:")
sentiment_results = sentiment_classifier(texts)
for text, result in zip(texts, sentiment_results):
    print(f"Text: {text}\nSentiment: {result['label']}, Confidence: {result['score']:.2f}\n")

# Named Entity Recognition (NER)
print("NER Results:")
ner_results = ner_classifier(texts)
for text, result in zip(texts, ner_results):
    print(f"Text: {text}")
    for entity in result:
        print(f"Entity: {entity['word']}, Type: {entity['entity_group']}, Confidence: {entity['score']:.2f}")
    print()
