"""
Project Title:
Generative Landscape Art Based on Environmental Themes

File Name:
generative_landscape_art_based_on_environmental_themes.py

Project Description:
In this project, we will generate landscape images inspired by environmental themes such as climate change, natural disasters, forest preservation, or pollution. The goal is to create stunning landscapes that evoke emotions related to these themes, making it a great step up in complexity by combining creative prompts with structured generation. We'll also add diversity to the images by varying the resolution and aspect ratio to match different artistic purposes (e.g., postcards, posters, or digital art). This exercise will test your understanding of crafting meaningful prompts while using the OpenAI API for more diverse image generations.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

def generate_landscape(prompt):
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

# Example use cases with environmental themes
prompts = [
    "A digital painting of a dense forest slowly being overtaken by pollution, with a gloomy, dark atmosphere.",
    "A serene coastal landscape experiencing rising sea levels, waves crashing against buildings with a sunset in the background.",
    "A futuristic city built with sustainable technology, surrounded by a green and thriving forest under a bright blue sky.",
    "A barren desert with cracked earth, dry rivers, and a few remaining trees showing the effects of extreme climate change.",
    "A lush mountain landscape during autumn, but the forest fire slowly approaching, creating a stark contrast between life and destruction."
]

# Generate images for all prompts
for prompt in prompts:
    print(f"Generating image for prompt: {prompt}")
    image_url = generate_landscape(prompt)
    if image_url:
        print(f"Image URL: {image_url}\n")
"""
Example Inputs and Expected Outputs:
Input Prompt:
"A digital painting of a dense forest slowly being overtaken by pollution, with a gloomy, dark atmosphere."
Expected Output:
An image of a dark, mysterious forest, with pollution in the air, and dead trees emerging, giving off a melancholy feeling.

Input Prompt:
"A serene coastal landscape experiencing rising sea levels, waves crashing against buildings with a sunset in the background."
Expected Output:
A coastal scene where water levels are rising, with houses partially submerged, and the sun setting over the horizon.

Input Prompt:
"A futuristic city built with sustainable technology, surrounded by a green and thriving forest under a bright blue sky."
Expected Output:
A modern, clean, eco-friendly city, with wind turbines, solar panels, and trees growing amidst skyscrapers under a vibrant sky.

Input Prompt:
"A barren desert with cracked earth, dry rivers, and a few remaining trees showing the effects of extreme climate change."
Expected Output:
A desolate desert with cracked ground, empty riverbeds, and a few scattered, struggling trees, conveying a sense of harshness and environmental decay.

Input Prompt:
"A lush mountain landscape during autumn, but the forest fire slowly approaching, creating a stark contrast between life and destruction."
Expected Output:
A picturesque autumn mountain scene with vibrant orange trees, contrasted with a fire slowly encroaching from one side, representing impending destruction.

This project elevates complexity by encouraging creative and environmental-focused prompts while generating diverse outputs with meaningful themes. The inputs are carefully designed to provoke thoughtful images reflecting pressing global issues.
"""