from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch
import os
import certifi

logging.set_verbosity_error()
# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()

# model_path = ("../../Models/models--mistralai--Mistral-7B-Instruct-v0.2/""snapshots/b70aa86578567ba3301b21c8a27bea4e8f6d6d61")

# model_path = "mistralai/Mistral-7B-Instruct-v0.2"  # ~29GB, restricted access!
# model_path = "KingNish/Reasoning-0.5b"  # (1) working! (bad results though)
# model_path = "arcee-ai/SuperNova-Medius" # too large (30GB - I had to try something else)
model_path = "jinaai/reader-lm-1.5b"  # (2) work!! (not very good results though)
# model_path = "deepseek-ai/DeepSeek-Coder-V2-Instruct" # too large
# model_path = "../models/models--bhadresh-savani--distilbert-base-uncased-emotion/snapshots/ce6f4ffcde7642ca2cac02381a16da38e5498ff7"  #didn't work!
# model_path = "../models/models--dbmdz--bert-large-cased-finetuned-conll03-english/snapshots/4c534963167c08d4b8ff1f88733cf2930f86add0"  # worked!! (not very good results though)
# model_path = "../models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13"  #didn't work!
# model_path = "../models/models--distilbert-base-cased-distilled-squad/snapshots/564e9b582944a57a3e586bbb98fd6f0a4118db7f"  # didn't work!
# model_path = "../models/models--facebook--bart-large-cnn/snapshots/37f520fa929c961707657b28798b30c003dd100b"  # bad result
# model_path = "../models/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d"  # bad result
# model_path = "../models/models--papluca--xlm-roberta-base-language-detection/snapshots/9865598389ca9d95637462f743f683b51d75b87b"  # bad result!
# model_path = "distilbert/distilgpt2" # very bad result
#model_path = "OpenAssistant/oasst-sft-1-pythia-12b"  # quite large! ~24GB
# model_path = "../models/Zyphra_Zamba2_2_7B_instruct"


device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.float16).to(device)

prompt = "give me an example of an animal that lives in water"

# case 1: if you are using a model that support chat like OpenAssistant or mistralai/Mistral-7B-Instruct
"""
messages = [{"role": "user", "content": prompt}]
encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
input_ids = encoded_input.to(device)
"""

# case 2: if you're doing simple text generation.
# Direct tokenization for basic text generation (no chat templates)
encoded_input = tokenizer(prompt, return_tensors="pt")
input_ids = encoded_input['input_ids'].to(device)

# Run Inference
output = model.generate(input_ids, max_new_tokens=100,
                        do_sample=True, top_p=0.95,
                        top_k=1000, temperature=0.2,
                        pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(output[0], skip_special_tokens=True))