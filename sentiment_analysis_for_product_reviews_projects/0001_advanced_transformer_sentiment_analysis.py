#!/usr/bin/env python3
"""
Advanced Transformer-based Sentiment Analysis for Product Reviews
Unique Reference: cddml-SrmZNuoOhMk
File: advanced_transformer_sentiment_analysis.py
"""

import os
import time
from typing import List, Dict, Any

import torch  # type: ignore
from torch import nn  # type: ignore
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)  # type: ignore
from datasets import load_dataset, Dataset  # type: ignore
import shap  # type: ignore

# Set device
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
MODEL_NAME: str = "distilbert-base-uncased"
NUM_LABELS: int = 3  # e.g. negative, neutral, positive


# 1. Data Loading and Preprocessing
def load_and_preprocess_data() -> Dataset:
    """
    Loads the 'amazon_polarity' dataset from Hugging Face Datasets,
    filters for product reviews, and maps labels to three classes.
    """
    dataset: Dict[str, Any] = load_dataset("amazon_polarity")
    # For demonstration, we assume a mapping: 0 -> negative, 1 -> positive, add neutral by sampling or smoothing.

    def map_labels(example: Dict[str, Any]) -> Dict[str, Any]:
        label: int = example["label"]
        # Introduce a 'neutral' class with a small probability for demonstration
        if torch.rand(1).item() < 0.1:
            example["label"] = 1  # assuming 1 as neutral for this demo; in practice, you’d have a better strategy.
        return example

    dataset["train"] = dataset["train"].map(map_labels)
    dataset["test"] = dataset["test"].map(map_labels)
    return dataset["train"]


# 2. Tokenization and Dataset Preparation
def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """
    Tokenizes the dataset using the provided tokenizer.
    """
    def tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        tokens: Dict[str, Any] = tokenizer(batch["content"], padding="max_length", truncation=True, max_length=128)
        return tokens

    tokenized_dataset: Dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset


# 3. Model Initialization and Training
def train_model(tokenized_dataset: Dataset) -> Trainer:
    """
    Fine-tunes a transformer model on the tokenized dataset.
    """
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)

    # Training arguments
    training_args: TrainingArguments = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(200)),  # using a small subset for evaluation demo
    )

    trainer.train()
    return trainer


# 4. Inference and Explainability
def analyze_sentiment(texts: List[str], model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Uses a Hugging Face pipeline for sentiment analysis.
    """
    nlp: Any = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0 if DEVICE=="cuda" else -1)
    results: List[Dict[str, Any]] = nlp(texts)
    return results


def explain_prediction(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer) -> None:
    """
    Generates a SHAP explanation for a single text input.
    """
    explainer: shap.Explainer = shap.Explainer(model, tokenizer, output_names=["Negative", "Neutral", "Positive"])
    shap_values: Any = explainer([text])
    shap.plots.text(shap_values[0])


# 5. Main Execution Function
def main() -> None:
    # Timestamp for logging and tracking
    timestamp: str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"Project run timestamp (UTC): {timestamp}")

    # Load and tokenize dataset
    print("Loading and preprocessing data...")
    raw_dataset: Dataset = load_and_preprocess_data()
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset: Dataset = tokenize_dataset(raw_dataset, tokenizer)

    # Train model
    print("Training model...")
    trainer: Trainer = train_model(tokenized_dataset)
    model: AutoModelForSequenceClassification = trainer.model
    model.to(DEVICE)

    # Save the model for deployment purposes
    model_save_path: str = "./advanced_sentiment_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Example use case demonstration
    example_texts: List[str] = [
        "The product quality was outstanding and delivery was prompt!",
        "I am extremely disappointed. The item broke after one use.",
        "It's okay, neither great nor terrible, just average."
    ]
    print("Analyzing sentiment for example texts...")
    results: List[Dict[str, Any]] = analyze_sentiment(example_texts, model, tokenizer)
    for text, res in zip(example_texts, results):
        print(f"Review: {text}\nPredicted Sentiment: {res}\n{'-'*50}")

    # Advanced: Explain one prediction with SHAP
    print("Generating explanation for the first example...")
    explain_prediction(example_texts[0], model, tokenizer)


if __name__ == "__main__":
    main()


# https://chatgpt.com/c/67dee0ae-e524-800c-84c0-d92418a458eb
