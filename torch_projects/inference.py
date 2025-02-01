import torch
from model_utils import load_model_and_tokenizer, encode_prompt

# Define model path and device
model_path = "../models/models--mistralai--Mistral-7B-Instruct-v0.2/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
tokenizer, model = load_model_and_tokenizer(model_path, device)

# Define a prompt for text generation
prompt = "Explain the theory of general relativity in simple terms."

# Encode the prompt using the tokenizer
input_ids = encode_prompt(tokenizer, prompt, device)

# Configure generation parameters
generation_params = {
    "max_new_tokens": 150,        # Maximum number of tokens to generate
    "do_sample": True,            # Enable sampling to make output more creative
    "top_p": 0.92,                # Top-p (nucleus sampling) controls diversity
    "top_k": 50,                  # Top-k sampling limits tokens considered at each step
    "temperature": 0.7,           # Softens probability distribution for more diverse outputs
    "pad_token_id": tokenizer.eos_token_id  # Pad with EOS token if necessary
}

# Generate the text based on the input prompt
output = model.generate(input_ids, **generation_params)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
