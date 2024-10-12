"""
Project Title:
"Multi-Concept Image Generation with Dynamic Prompts"

File Name:
multi_concept_image_generation_with_dynamic_prompts.py

Project Description:
This project focuses on generating images based on multiple dynamic concepts, combining objects, environments, and artistic styles. You will create a system that generates diverse image prompts using a combination of predefined concepts. The program will then send these prompts to the OpenAI image generation API and retrieve the generated images.

The aim of the project is to:

Dynamically combine different subjects (e.g., animals, landscapes, objects) with various artistic styles.
Allow the user to specify the combination or use random selection for each request.
Automate the generation of multiple images in different styles and concepts using loops or batch requests.
Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # Import API key from local file
import random

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Define concepts for dynamic image generation
subjects = ['a futuristic city', 'a peaceful forest', 'an alien landscape', 'a surreal dream', 'a mythical creature']
art_styles = ['in the style of impressionism', 'with hyper-realistic details', 'as a 3D render', 'in watercolors',
              'as a pencil sketch']


# Function to generate images based on dynamic prompts
def generate_image_from_prompt():
    # Randomly combine a subject and an art style
    subject = random.choice(subjects)
    style = random.choice(art_styles)

    # Generate the prompt by combining the subject and style
    prompt = f"An image of {subject}, {style}."
    print(f"Generating image with prompt: {prompt}")

    # Send the prompt to the OpenAI image generation API
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # Fixed size for all generated images
    )

    # Extract the generated image URL
    image_url = response.data[0].url
    print(f"Generated Image URL: {image_url}")
    return image_url


# Generate multiple images (5 for this example)
if __name__ == "__main__":
    for i in range(5):
        generate_image_from_prompt()
"""
Example Inputs and Expected Outputs:
Input: "An image of a futuristic city, in the style of impressionism." Expected Output: A futuristic city painted with impressionistic brushstrokes, soft and abstract in appearance.

Input: "An image of a peaceful forest, with hyper-realistic details." Expected Output: A detailed forest scene with vivid, lifelike textures and lighting.

Input: "An image of an alien landscape, as a 3D render." Expected Output: A highly detailed 3D-rendered alien planet with futuristic elements and unusual terrain.

Input: "An image of a surreal dream, in watercolors." Expected Output: A dream-like abstract scene with soft, flowing watercolor strokes and blended colors.

Input: "An image of a mythical creature, as a pencil sketch." Expected Output: A detailed pencil drawing of a fantastical creature, showcasing intricate line work and shading.

Summary:
This exercise combines dynamic prompt generation with OpenAI’s image generation capabilities, allowing the user to generate a wide range of conceptually diverse images. The random selection of subjects and styles adds variation, making the project adaptable for further exploration and experimentation.
"""