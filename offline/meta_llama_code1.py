import torch
from transformers import pipeline
import certifi
import os
# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()
model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

pipe("The key to life is")