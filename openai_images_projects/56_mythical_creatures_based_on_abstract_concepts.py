"""
Project Title: AI-Generated Mythical Creatures Based on Abstract Concepts
File Name:
mythical_creatures_based_on_abstract_concepts.py

Description:
In this exercise, we will generate images of mythical creatures inspired by abstract concepts such as "chaos," "serenity," and "transcendence." The aim is to create highly creative representations of mythical beings that visually embody these abstract ideas. This project will push your use of OpenAI's image generation model to explore advanced themes by providing rich, imaginative prompts.

Python Code:

python
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

# Example use cases
if __name__ == "__main__":
    prompts = [
        "A chaotic, multi-headed dragon made of swirling fire and shadows, representing the concept of chaos",
        "A phoenix with feathers made of light, soaring peacefully over a still lake, symbolizing serenity",
        "A winged lion with glowing eyes, walking through an abstract realm filled with shifting geometric shapes, representing transcendence",
        "A griffin with crystalline wings, flying through a sky filled with shimmering stars, embodying the idea of wonder",
        "A sea serpent made of mist and smoke, coiling through a dreamlike ocean, representing the concept of illusion"
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
Prompt: "A chaotic, multi-headed dragon made of swirling fire and shadows, representing the concept of chaos"
Expected Output:
A generated image URL featuring a dragon with multiple heads, surrounded by swirling fire and shadows, visually representing chaos.

Input:
Prompt: "A phoenix with feathers made of light, soaring peacefully over a still lake, symbolizing serenity"
Expected Output:
A generated image URL depicting a glowing phoenix peacefully flying over a calm, reflective lake.

Input:
Prompt: "A winged lion with glowing eyes, walking through an abstract realm filled with shifting geometric shapes, representing transcendence"
Expected Output:
A generated image URL showing a mystical lion walking through an abstract world of changing geometric forms.

Input:
Prompt: "A griffin with crystalline wings, flying through a sky filled with shimmering stars, embodying the idea of wonder"
Expected Output:
A generated image URL illustrating a griffin soaring through a starlit sky with transparent, crystal-like wings.

Input:
Prompt: "A sea serpent made of mist and smoke, coiling through a dreamlike ocean, representing the concept of illusion"
Expected Output:
A generated image URL depicting a misty sea serpent twisting through a surreal, dreamlike sea, evoking a sense of illusion.
"""