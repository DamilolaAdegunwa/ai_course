from transformers import AutoModelForCausalLM, AutoTokenizer
import certifi
import os
# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
MAX_REASONING_TOKENS = 1024
MAX_RESPONSE_TOKENS = 512

model_name = "KingNish/Reasoning-Llama-1b-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Which is greater 9.9 or 9.11 ??"
messages = [
    {"role": "user", "content": prompt}
]

# Generate reasoning
reasoning_template = tokenizer.apply_chat_template(messages, tokenize=False, add_reasoning_prompt=True)
reasoning_inputs = tokenizer(reasoning_template, return_tensors="pt").to(model.device)
reasoning_ids = model.generate(**reasoning_inputs, max_new_tokens=MAX_REASONING_TOKENS)
reasoning_output = tokenizer.decode(reasoning_ids[0, reasoning_inputs.input_ids.shape[1]:], skip_special_tokens=True)

# print("REASONING: " + reasoning_output)

# Generate answer
messages.append({"role": "reasoning", "content": reasoning_output})
response_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
response_inputs = tokenizer(response_template, return_tensors="pt").to(model.device)
response_ids = model.generate(**response_inputs, max_new_tokens=MAX_RESPONSE_TOKENS)
response_output = tokenizer.decode(response_ids[0, response_inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("ANSWER: " + response_output)

"""
working!!!
"""
