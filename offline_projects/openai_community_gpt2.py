from transformers import pipeline, set_seed, logging
import certifi
import os
# Logging activities with priority of info and above
logging.set_verbosity_info()
# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
generator = pipeline('text-generation', model='gpt2', device=0)
# set_seed(42)
# generator("Hello, I'm a language model,", max_length=300, num_return_sequences=5, truncation=True, pad_token_id=77)
generator("Explain what Transformer is in Gen AI")