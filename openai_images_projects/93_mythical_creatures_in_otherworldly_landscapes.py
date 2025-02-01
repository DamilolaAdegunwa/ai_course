"""
Project Title: Mythical Creatures in Otherworldly Landscapes
File Name: mythical_creatures_in_otherworldly_landscapes.py
Description:
This project focuses on generating images of mythical creatures set against surreal, otherworldly landscapes. The creatures can range from dragons, phoenixes, and centaurs to entirely new beings inspired by fantasy lore. These beings will inhabit vast and unique environments—whether in alien planets, enchanted forests, or magical realms. The purpose of this exercise is to explore how OpenAI’s image generation can handle detailed, imaginative compositions by blending creatures with their exotic habitats.

You will use prompts to create highly detailed and fantastical scenarios, showcasing both creature design and landscape artistry. This project will challenge your ability to craft prompts that weave together biological, environmental, and artistic details into cohesive and visually striking AI-generated images.

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


# Function to generate an image of a mythical creature in an otherworldly landscape
def generate_creature_image(prompt):
    """
    Generate an image of a mythical creature in an otherworldly landscape using OpenAI's image generation API.

    :param prompt: The prompt describing the creature and the landscape.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed fantasy worlds and creatures
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
    # Define multiple prompts involving mythical creatures and otherworldly landscapes
    creature_prompts = [
        "A phoenix flying over a volcano in an alien desert with three suns in the sky",
        "A dragon resting on a cliff in a magical forest with floating crystals and glowing plants",
        "A centaur battling a shadow creature in a fog-covered forest surrounded by ancient ruins",
        "A giant griffin soaring through an ethereal sky with floating islands and waterfalls made of light",
        "A majestic unicorn standing on a hill in a dreamlike meadow with swirling clouds and rainbow trees"
    ]

    # Generate, download, and save each creature image
    for i, prompt in enumerate(creature_prompts):
        print(f"Generating mythical creature image for prompt: '{prompt}'...")
        creature_url = generate_creature_image(prompt)
        creature_image = download_image(creature_url)
        save_image(creature_image, f"mythical_creature_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A phoenix flying over a volcano in an alien desert with three suns in the sky"
Expected Output:
An image of a bright, fiery phoenix flying high above an active volcano. The landscape is barren and desert-like, with three suns casting an intense glow across the scene.
Input:

Prompt: "A dragon resting on a cliff in a magical forest with floating crystals and glowing plants"
Expected Output:
A scene of a powerful dragon perched on a cliff, overlooking a mystical forest. The trees are bioluminescent, and crystals float in mid-air, illuminating the forest with a magical light.
Input:

Prompt: "A centaur battling a shadow creature in a fog-covered forest surrounded by ancient ruins"
Expected Output:
A tense, dynamic image of a centaur locked in combat with a shadowy figure in a forest thick with fog. Ancient ruins and broken columns can be seen in the background, adding to the eerie atmosphere.
Input:

Prompt: "A giant griffin soaring through an ethereal sky with floating islands and waterfalls made of light"
Expected Output:
A majestic griffin flies through an ethereal sky where islands float above the clouds, and waterfalls made of glowing light cascade down from them, creating a surreal and heavenly atmosphere.
Input:

Prompt: "A majestic unicorn standing on a hill in a dreamlike meadow with swirling clouds and rainbow trees"
Expected Output:
A serene and magical scene of a unicorn standing gracefully on a hill. The meadow is filled with vibrant, rainbow-colored trees, and the sky swirls with fantastical clouds, giving the entire scene a dreamlike quality.
Project Overview:
This project allows you to explore the limits of creativity and storytelling through OpenAI’s image generation, with a focus on combining mythical creatures and otherworldly environments. By experimenting with different prompts that blend the fantastical and the surreal, you will enhance your ability to craft highly detailed and imaginative visuals. The result will be a set of powerful, evocative images that capture the spirit of fantasy art.







"""