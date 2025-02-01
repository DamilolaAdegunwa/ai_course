"""
Project Title: Generating Futuristic Cityscapes with Environmental Effects
File Name:
generating_futuristic_cityscapes_with_environmental_effects.py

Description: In this project, you will generate futuristic cityscapes with various environmental effects such as fog, rain, glowing lights, and reflections. These scenes focus on advanced urban environments, incorporating weather and atmospheric effects to create unique visuals. The user provides prompts with detailed descriptions, and the model generates high-quality images based on those inputs.

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
        "A futuristic city at night with glowing neon lights, thick fog, and reflections on wet streets",
        "A massive futuristic metropolis during a rainstorm, with towering skyscrapers and flying vehicles",
        "A cyberpunk cityscape at dusk, where holographic billboards hover over the streets and light mist fills the air",
        "A futuristic coastal city with a glass-like sea reflecting the shimmering city lights under a starry sky",
        "A dense futuristic city covered in smog, with dim lights barely penetrating the haze"
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
Prompt: "A futuristic city at night with glowing neon lights, thick fog, and reflections on wet streets"
Expected Output:
A generated image URL showcasing a dark city with bright neon lights, dense fog, and wet, reflective streets.

Input:
Prompt: "A massive futuristic metropolis during a rainstorm, with towering skyscrapers and flying vehicles"
Expected Output:
A generated image URL featuring a sprawling futuristic city with towering buildings, heavy rain, and vehicles flying between skyscrapers.

Input:
Prompt: "A cyberpunk cityscape at dusk, where holographic billboards hover over the streets and light mist fills the air"
Expected Output:
A generated image URL depicting a cyberpunk-style city at dusk, with holograms floating in the air, and a misty atmosphere.

Input:
Prompt: "A futuristic coastal city with a glass-like sea reflecting the shimmering city lights under a starry sky"
Expected Output:
A generated image URL showing a futuristic coastal city at night, with calm, glassy water reflecting city lights and a star-filled sky.

Input:
Prompt: "A dense futuristic city covered in smog, with dim lights barely penetrating the haze"
Expected Output:
A generated image URL illustrating a heavily polluted city where smog obscures buildings, and weak city lights struggle to cut through the haze.







"""