"""
Project Title: Generating Mythical Creatures in Surreal Environments
File Name:
generating_mythical_creatures_in_surreal_environments.py

Description: In this project, you will generate images of mythical creatures such as dragons, phoenixes, and unicorns, set in surreal, otherworldly environments. The scenes will incorporate unusual settings like floating islands, glowing forests, or alien landscapes. The prompts will be rich with imaginative detail, and the model will generate visually stunning images based on these inputs.

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
        "A phoenix rising from glowing lava on a floating island surrounded by storm clouds",
        "A massive dragon with molten scales resting in a valley of luminescent crystals under an alien sky",
        "A unicorn galloping through a mystical forest where the trees are bioluminescent and the ground is made of glass",
        "A gargantuan sea serpent emerging from an ocean made of liquid silver, surrounded by glowing mist",
        "A celestial griffin flying through the air above a world made entirely of swirling colors and floating rocks"
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
Prompt: "A phoenix rising from glowing lava on a floating island surrounded by storm clouds"
Expected Output:
A generated image URL featuring a glowing phoenix rising from molten lava on a floating island, with stormy clouds surrounding the scene.

Input:
Prompt: "A massive dragon with molten scales resting in a valley of luminescent crystals under an alien sky"
Expected Output:
A generated image URL showing a dragon with glowing scales resting in a vibrant, crystal-covered valley beneath a strange, colorful sky.

Input:
Prompt: "A unicorn galloping through a mystical forest where the trees are bioluminescent and the ground is made of glass"
Expected Output:
A generated image URL depicting a unicorn running through a surreal forest with glowing trees and a reflective, glassy surface.

Input:
Prompt: "A gargantuan sea serpent emerging from an ocean made of liquid silver, surrounded by glowing mist"
Expected Output:
A generated image URL of a huge sea serpent rising from a silvery ocean, with mist glowing in the background.

Input:
Prompt: "A celestial griffin flying through the air above a world made entirely of swirling colors and floating rocks"
Expected Output:
A generated image URL showing a griffin soaring over a psychedelic landscape of swirling colors and floating stones.
"""
