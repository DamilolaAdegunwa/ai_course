# openai_community_gpt2c.py
from transformers import GPT2Tokenizer, TFGPT2Model
import certifi
import os
# Logging activities with priority of info and above
# logging.set_verbosity_info()
# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
