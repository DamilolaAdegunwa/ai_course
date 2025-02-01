"""
Project Title:
Dynamic Prompt Enrichment for Enhanced Image Generation

File Name:
dynamic_prompt_image_generation.py

Project Description:
In this project, you will create a Python script that takes basic user prompts and dynamically enriches them by adding stylistic elements, emotions, and descriptive adjectives. This enriched prompt is then used to generate more detailed and imaginative images using OpenAI's image generation API. By utilizing predefined lists of styles, emotions, and descriptors, the script enhances creativity and produces a wide variety of artistic outputs.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # The file containing the API key
import random

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def enrich_prompt(basic_prompt):
    """Enriches the basic prompt with random styles, emotions, and descriptors."""
    styles = [
        "in the style of Van Gogh",
        "as a digital art piece",
        "like a watercolor painting",
        "as a 3D render",
        "in a minimalist style"
    ]
    emotions = [
        "with a joyful mood",
        "with a melancholic atmosphere",
        "with energetic vibes",
        "evoking serenity",
        "with a mysterious aura"
    ]
    descriptors = [
        "vibrant colors",
        "soft lighting",
        "high contrast",
        "monochromatic tones",
        "dynamic composition"
    ]

    style = random.choice(styles)
    emotion = random.choice(emotions)
    descriptor = random.choice(descriptors)

    enriched_prompt = f"{basic_prompt}, {descriptor}, {emotion}, {style}"
    return enriched_prompt


def generate_image_from_prompt(prompt):
    """Generates an image from an enriched prompt."""
    # Generate the image based on the enriched prompt
    response = client.images.generate(
        prompt=prompt,
        size="512x512",  # You can use "256x256", "512x512", or "1024x1024"
        n=1  # Number of images to generate
    )
    # Extract and return the image URL
    return response.data[0].url


# Example usage:
if __name__ == "__main__":
    basic_prompts = [
        "A serene forest landscape",
        "An ancient castle on a hill",
        "A futuristic robot companion",
        "An exotic tropical bird",
        "A mysterious floating island"
    ]

    for i, basic_prompt in enumerate(basic_prompts):
        print(f"Generating image {i + 1}...")
        enriched_prompt = enrich_prompt(basic_prompt)
        print(f"Enriched Prompt: {enriched_prompt}")
        image_url = generate_image_from_prompt(enriched_prompt)
        print(f"Image {i + 1} URL: {image_url}\n")
"""
Example Inputs and Expected Outputs:
Input:
Basic Prompt: A serene forest landscape
Enriched Prompt (example): A serene forest landscape, vibrant colors, with a mysterious aura, in the style of Van Gogh
Expected Output:
A URL linking to an image depicting a forest landscape with vibrant colors, a mysterious feeling, and styled similar to Van Gogh's artwork.

Input:
Basic Prompt: An ancient castle on a hill
Enriched Prompt (example): An ancient castle on a hill, high contrast, with energetic vibes, as a 3D render
Expected Output:
A URL linking to an image of a castle rendered in 3D, with high contrast and an energetic atmosphere.

Input:
Basic Prompt: A futuristic robot companion
Enriched Prompt (example): A futuristic robot companion, monochromatic tones, evoking serenity, in a minimalist style
Expected Output:
A URL linking to an image of a robot designed with minimalist elements, monochromatic colors, and a serene mood.

Input:
Basic Prompt: An exotic tropical bird
Enriched Prompt (example): An exotic tropical bird, soft lighting, with a joyful mood, like a watercolor painting
Expected Output:
A URL linking to an image of a tropical bird painted in watercolor style, with soft lighting and a joyful atmosphere.

Input:
Basic Prompt: A mysterious floating island
Enriched Prompt (example): A mysterious floating island, dynamic composition, with a melancholic atmosphere, as a digital art piece
Expected Output:
A URL linking to a digital art image of a floating island with dynamic composition and a melancholic feel.

This project enhances your previous work by introducing dynamic prompt enrichment, which adds randomness and creativity to the image generation process. By combining various styles, emotions, and descriptors, you can explore a vast array of artistic possibilities and see how different prompt variations influence the generated images.
"""
