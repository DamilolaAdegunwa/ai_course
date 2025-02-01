"""
https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833
Project Title: Mythical Creatures in Surreal Landscapes
File Name: mythical_creatures_in_surreal_landscapes.py
Description:
This project focuses on creating surreal landscapes populated by mythical creatures, such as dragons, phoenixes, or unicorns, using detailed AI-generated imagery. The project will allow you to explore combining highly imaginative prompts with fantastical elements, pushing your creativity and precision in generating distinct, rich visuals.

You'll create mythical creatures with unique characteristics and place them in surreal environments, such as floating islands, lava-filled realms, or bioluminescent forests. This advanced task will challenge your prompt-engineering skills and introduce multiple creative possibilities.

Python Code:
"""
import os
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate mythical creatures in surreal landscapes
def generate_mythical_image(prompt):
    """
    Generate an image based on a mythical creature and surreal landscape prompt using OpenAI's image generation API.

    :param prompt: The prompt describing the mythical creature and its surreal environment.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed imagery
    )

    return response.data[0].url  # Returns the URL of the generated image


# Function to download the image from a URL and return it as a PIL Image object
def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: The URL of the image.
    :return: PIL Image object
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


# Function to save the image to a file
def save_image(image, filename):
    """
    Save the downloaded image to a file.

    :param image: PIL Image object.
    :param filename: The path and name of the file to save the image to.
    """
    image.save(filename)
    print(f"Image saved as {filename}")


# Example use cases
if __name__ == "__main__":
    # Define multiple prompts involving mythical creatures in surreal landscapes
    creature_prompts = [
        "A giant dragon soaring above floating islands, with waterfalls cascading into the void below",
        "A glowing phoenix flying over a bioluminescent forest, illuminating the trees with fiery wings",
        "A unicorn standing atop a mountain made of crystal, with the northern lights illuminating the sky",
        "A kraken emerging from an ocean of lava, surrounded by black volcanic mountains",
        "A griffin flying above a city in the clouds, with golden towers and sky bridges connecting floating castles"
    ]

    # Generate, download, and save each mythical creature image
    for i, prompt in enumerate(creature_prompts):
        print(f"Generating mythical image for prompt: '{prompt}'...")
        creature_url = generate_mythical_image(prompt)
        creature_image = download_image(creature_url)
        save_image(creature_image, f"mythical_creature_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A giant dragon soaring above floating islands, with waterfalls cascading into the void below"
Expected Output:
An image of a majestic dragon flying over floating islands suspended in the sky. Waterfalls pour off the edges of the islands, disappearing into the misty void below.
Input:

Prompt: "A glowing phoenix flying over a bioluminescent forest, illuminating the trees with fiery wings"
Expected Output:
A vibrant phoenix with glowing, fiery wings flying above a surreal bioluminescent forest. The forest floor glows in shades of blue and green, with the phoenix lighting up the landscape as it passes overhead.
Input:

Prompt: "A unicorn standing atop a mountain made of crystal, with the northern lights illuminating the sky"
Expected Output:
A sparkling unicorn standing proudly on a mountain made entirely of transparent crystal. Above it, the northern lights dance across the night sky in vibrant waves of color.
Input:

Prompt: "A kraken emerging from an ocean of lava, surrounded by black volcanic mountains"
Expected Output:
A terrifying kraken with glowing, molten tentacles rising from a sea of lava. Volcanic mountains, dark and ominous, surround the creature, with rivers of lava flowing down their slopes.
Input:

Prompt: "A griffin flying above a city in the clouds, with golden towers and sky bridges connecting floating castles"
Expected Output:
A majestic griffin flying through the air above a city in the clouds. The city consists of floating castles with golden towers, connected by sky bridges that seem to defy gravity.
Project Overview:
This exercise will take your skills to the next level by focusing on the generation of mythical creatures within surreal and fantastical landscapes. You will explore the limits of prompt engineering to design unique scenes involving mythical beings and their environments. This project allows for deep creativity, testing your ability to craft rich visual prompts and output highly detailed artwork in various mythical settings.







"""