import os
from transformers import pipeline

# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = ''

# Now you can call the pipeline without SSL verification
classifier = pipeline(task="text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(classifier(["You are the best", "Get Lost"]))
