from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model_and_tokenizer(model_path, device):
    """
    Load the tokenizer and model from the given path and prepare for inference.

    Args:
    - model_path (str): Path to the pre-trained model checkpoint.
    - device (str): Device to run the model ('cuda' or 'cpu').

    Returns:
    - tokenizer: Loaded tokenizer object.
    - model: Loaded causal language model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    return tokenizer, model


def encode_prompt(tokenizer, prompt, device):
    """
    Encode the prompt for the model using the tokenizer.

    Args:
    - tokenizer: Pre-trained tokenizer.
    - prompt (str): The input text prompt.
    - device (str): Device to run the model.

    Returns:
    - input_ids: Tokenized input ready for inference.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    return input_ids.to(device)
