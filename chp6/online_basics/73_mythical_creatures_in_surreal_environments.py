"""
Project Title: Mythical Creatures in Surreal Environments
File Name: mythical_creatures_in_surreal_environments.py
Description:
This project explores the creation of mythical creatures set within surreal, otherworldly environments. You'll generate unique visual representations of creatures like dragons, phoenixes, and griffins in dreamlike landscapes. The exercise focuses on complex prompt engineering to blend detailed creature descriptions with fantastical environments, allowing for creative storytelling through imagery.

This is a noticeable step up in difficulty as it challenges you to combine both character design (mythical creatures) and environmental elements in a way that evokes a sense of magic and mystery.

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


# Function to generate images of mythical creatures in surreal environments
def generate_mythical_image(prompt):
    """
    Generate an image based on a mythical creature and a surreal environment using OpenAI's image generation API.

    :param prompt: The prompt describing the mythical creature and surreal environment.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # Larger size for capturing detailed surreal environments
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
    # Define multiple prompts involving mythical creatures in surreal environments
    creature_prompts = [
        "A glowing phoenix rising from a lake of liquid gold surrounded by misty mountains",
        "A giant serpent coiled around a floating island with waterfalls cascading into the void",
        "A griffin soaring above a crimson sky filled with ethereal clouds and twin moons",
        "A dragon curled up in a crystalline cave with glowing mushrooms and a waterfall of light",
        "A unicorn standing in an enchanted forest where trees are made of glass and the ground glows softly"
    ]

    # Generate, download, and save each mythical creature image
    for i, prompt in enumerate(creature_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_mythical_image(prompt)
        image = download_image(image_url)
        save_image(image, f"mythical_creature_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A glowing phoenix rising from a lake of liquid gold surrounded by misty mountains"
Expected Output:
An image of a majestic phoenix glowing with flames, emerging from a shimmering lake of molten gold. The background consists of ethereal, mist-covered mountains.
Input:

Prompt: "A giant serpent coiled around a floating island with waterfalls cascading into the void"
Expected Output:
A surreal scene of a massive serpent coiled around a floating island suspended in mid-air. Waterfalls flow endlessly off the edge of the island, disappearing into the infinite void below.
Input:

Prompt: "A griffin soaring above a crimson sky filled with ethereal clouds and twin moons"
Expected Output:
A powerful griffin flying across a red-tinted sky, with two large moons visible in the background. The clouds are dreamlike and seem to glow faintly with a mystical aura.
Input:

Prompt: "A dragon curled up in a crystalline cave with glowing mushrooms and a waterfall of light"
Expected Output:
A dark yet magical scene of a dragon resting in a cave made entirely of crystal. The cave is illuminated by bioluminescent mushrooms and a surreal waterfall made of glowing light.
Input:

Prompt: "A unicorn standing in an enchanted forest where trees are made of glass and the ground glows softly"
Expected Output:
A peaceful image of a unicorn in an otherworldly forest where the trees are crafted from transparent glass, and the forest floor emits a soft, magical glow.
Project Overview:
This project helps you build expertise in merging vivid descriptions of creatures and environments to create images that tell a story. It challenges your ability to convey complexity and uniqueness in both character (mythical creatures) and setting (surreal environments), taking you a step closer to mastering creative prompt engineering. Through these exercises, you will produce high-quality imagery with strong fantasy and surreal elements.







"""