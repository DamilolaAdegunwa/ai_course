from transformers import GPT2Tokenizer, GPT2Model, logging
import certifi
import os
# Logging activities with priority of info and above
# logging.set_verbosity_info()
# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Explain GPTs"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
