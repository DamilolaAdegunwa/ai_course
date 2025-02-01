import os
from transformers import pipeline
import certifi

# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()

# Now you can call the pipeline without SSL verification
classifier = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(classifier(["You are the best", "Get Lost"]))
