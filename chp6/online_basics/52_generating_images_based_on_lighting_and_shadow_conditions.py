"""
Project Title: Generating Images Based on Complex Lighting and Shadow Conditions
File Name:
generating_images_based_on_lighting_and_shadow_conditions.py

Description: In this exercise, we will explore how to generate images based on complex lighting and shadow conditions. The user will provide a prompt that includes details of the lighting setup (e.g., "morning sunlight casting long shadows", "a room lit by a single candle", or "neon lights reflecting on a wet street"). The AI will then create an image that represents these conditions, considering how light interacts with objects in the scene.

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
        "A forest at dawn with sunlight filtering through the trees, casting long shadows on the ground",
        "A futuristic city street at night, with neon lights reflecting on the wet pavement",
        "A living room lit by a single candle, creating deep shadows and warm light",
        "A snowy landscape at dusk, with a soft glow on the snow and long shadows from the trees",
        "A desert at noon with harsh sunlight and minimal shadows due to the intense light"
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
Prompt: "A forest at dawn with sunlight filtering through the trees, casting long shadows on the ground"
Expected Output:
A generated image URL showing a forest scene with soft, early morning light streaming through the trees and long shadows across the forest floor.

Input:
Prompt: "A futuristic city street at night, with neon lights reflecting on the wet pavement"
Expected Output:
A generated image URL showing a futuristic city street at night, with vivid neon signs and reflections on the slick, wet ground.

Input:
Prompt: "A living room lit by a single candle, creating deep shadows and warm light"
Expected Output:
A generated image URL of a cozy living room dimly lit by a candle, with soft shadows and warm, flickering light.

Input:
Prompt: "A snowy landscape at dusk, with a soft glow on the snow and long shadows from the trees"
Expected Output:
A generated image URL of a snow-covered scene at dusk, with gentle light reflecting off the snow and long shadows cast by the trees.

Input:
Prompt: "A desert at noon with harsh sunlight and minimal shadows due to the intense light"
Expected Output:
A generated image URL of a barren desert landscape at midday, with harsh, bright sunlight and almost no visible shadows due to the overhead sun.
"""