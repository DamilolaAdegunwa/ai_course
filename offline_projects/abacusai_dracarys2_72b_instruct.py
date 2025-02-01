import transformers
import torch
import certifi
import os
# Disable SSL verification globally
os.environ['CURL_CA_BUNDLE'] = certifi.where()

model_id = "abacusai/Dracarys2-72B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are data science coding assistant that generates Python code using Pandas and Numpy."},
    {"role": "user", "content": "Write code to select rows from the dataframe `df` having the maximum `temp` for each `city`"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

"""
stopped it because it is about 160GB 😭😭😭

The "abacusai/Dracarys2-72B-Instruct" model is quite large, with approximately 72.7 billion parameters. Typically, models of this size require significant storage space. For example, individual components of the model (like specific checkpoint files) can be around 5 GB each, and the entire model may span multiple files. In total, the model is estimated to take up around 140-160 GB of space when fully downloaded and ready for use​
HUGGING FACE
​
HUGGING FACE
.

Ensure you have enough storage and memory resources if you plan on working with it!
"""