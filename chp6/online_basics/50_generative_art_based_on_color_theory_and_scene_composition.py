"""
https://chatgpt.com/c/66f9e677-b2c4-800c-9ad8-f3b4301249bb
Project Title: Advanced Generative Art Based on Color Theory and Scene Composition
Description: In this exercise, we will create an advanced OpenAI image generation project that incorporates concepts of color theory and scene composition. The user will provide a prompt that includes a specific color palette and a type of scene (e.g., landscape, futuristic city, etc.). The project will generate images based on the prompt and return the image URL.

Python Code:
"""

import os
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

def generate_image_from_prompt(prompt):
    try:
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example use cases:
if __name__ == "__main__":
    prompts = [
        "A serene landscape at sunset with a pastel color palette",
        "A futuristic cityscape with neon colors and flying cars",
        "An underwater scene with coral reefs and deep blue tones",
        "A cozy living room with warm earthy colors and natural light",
        "A mountain range at dawn with a gradient of purples and pinks in the sky"
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        image_url = generate_image_from_prompt(prompt)
        if image_url:
            print(f"Generated Image URL: {image_url}\n")
        else:
            print("Failed to generate image.\n")

"""
Example Input(s) with Expected Output(s):
Input:
Prompt: "A serene landscape at sunset with a pastel color palette"
Expected Output:
A generated image URL for a serene landscape scene, featuring soft, pastel hues in the sky and landscape.

Input:
Prompt: "A futuristic cityscape with neon colors and flying cars"
Expected Output:
A generated image URL for a bright, neon-lit cityscape with flying vehicles and a futuristic skyline.

Input:
Prompt: "An underwater scene with coral reefs and deep blue tones"
Expected Output:
A generated image URL showcasing an underwater environment with vibrant coral reefs and shades of blue.

Input:
Prompt: "A cozy living room with warm earthy colors and natural light"
Expected Output:
A generated image URL of a cozy living room scene, with soft lighting and warm tones in the furniture and decor.

Input:
Prompt: "A mountain range at dawn with a gradient of purples and pinks in the sky"
Expected Output:
A generated image URL of a beautiful mountain range at dawn, with the sky transitioning from purples to pinks.

File Name:
generative_art_based_on_color_theory_and_scene_composition.py
"""