import os
from transformers import pipeline, logging
import certifi

logging.set_verbosity(logging.WARNING)
model_path = ("../models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/"
              "snapshots/714eb0fa89d2f80546fda750413ed43d93601a13")

# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['ENV'] = 'dev'
os.environ['ENVIRONMENT'] = 'dev'

# Now you can call the pipeline without SSL verification
classifier = pipeline(task="text-classification", device=0, model=model_path)
print(classifier(["You are the best", "Get Lost"]))
