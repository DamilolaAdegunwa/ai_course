"""
Project Title: Fantasy Creature Creation
File Name: fantasy_creature_creation.py
Description:
In this project, you'll explore the fascinating world of fantasy creatures by generating unique and imaginative designs based on specific characteristics provided through prompts. You will create a program that allows users to specify traits such as size, color, habitat, and special abilities, resulting in a one-of-a-kind creature generated through the OpenAI API. This project not only enhances your understanding of image generation but also encourages creative thinking and descriptive articulation.

Python Code:
"""
import os
from io import BytesIO

from PIL.Image import Image
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key
import requests

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate fantasy creature images based on user specifications
def generate_creature_image(size, color, habitat, ability):
    """
    Generate an image of a fantasy creature based on the given characteristics.

    :param size: The size of the creature (e.g., small, medium, large).
    :param color: The primary color of the creature.
    :param habitat: The habitat where the creature lives (e.g., forest, ocean, desert).
    :param ability: A special ability of the creature (e.g., flying, invisibility).
    :return: URL of the generated creature image.
    """
    prompt = f"A {size} {color} creature that lives in the {habitat} and has the ability to {ability}."
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution for detailed images
    )

    return response.data[0].url  # Returns the URL of the generated image


# Function to display and save the generated creature image
def display_and_save_image(image_url, filename):
    """
    Download and save the creature image to a file.

    :param image_url: The URL of the generated creature image.
    :param filename: The path and name of the file to save the image to.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img.save(filename)
    print(f"Image saved as {filename}")


# Example use cases
if __name__ == "__main__":
    # Define multiple creature characteristics
    creature_specs = [
        ("small", "blue", "ocean", "swim fast"),
        ("large", "green", "forest", "camouflage"),
        ("medium", "red", "mountains", "fly"),
        ("tiny", "purple", "desert", "create mirages"),
        ("gigantic", "black", "cave", "breath fire")
    ]

    # Generate and save images for each creature specification
    for specs in creature_specs:
        size, color, habitat, ability = specs
        print(f"Generating image for a {size} {color} creature...")
        image_url = generate_creature_image(size, color, habitat, ability)
        display_and_save_image(image_url, f"fantasy_creature_{size}_{color}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Size: "small"
Color: "blue"
Habitat: "ocean"
Ability: "swim fast"
Expected Output:
An image of a small blue creature that resembles a sea creature, perhaps with fins and a streamlined body, capable of swift swimming.
Input:

Size: "large"
Color: "green"
Habitat: "forest"
Ability: "camouflage"
Expected Output:
An image of a large green creature, resembling a reptile or a mammal, blending in with the foliage of a forest.
Input:

Size: "medium"
Color: "red"
Habitat: "mountains"
Ability: "fly"
Expected Output:
An image of a medium-sized red creature with wings, soaring above rocky mountain peaks.
Input:

Size: "tiny"
Color: "purple"
Habitat: "desert"
Ability: "create mirages"
Expected Output:
An image of a tiny purple creature, perhaps resembling an insect, surrounded by shimmering heat waves in a desert.
Input:

Size: "gigantic"
Color: "black"
Habitat: "cave"
Ability: "breath fire"
Expected Output:
An image of a gigantic black creature, similar to a dragon, lurking in the shadows of a cave, exhaling flames.
Project Overview:
This project challenges you to think creatively and descriptively about fantasy creatures while leveraging the power of AI-generated imagery. Each generated creature will serve as a testament to your imaginative input, and the resulting images can be used for storytelling, game design, or personal enjoyment. Embrace the limitless possibilities of your creativity and have fun with the fantasy realm!
"""