"""
Project Title: Creating a Fantasy Scene Generator with Background Elements
Description:
In this project, you will create a program that generates a fantasy scene composed of various elements, such as characters, landscapes, and magical items. You will combine several generated images to create a rich and immersive scene. This project is designed to enhance your skills in using OpenAI's image generation capabilities by integrating multiple elements into one cohesive image.

This will involve generating different elements like:

A fantasy character (e.g., wizard, warrior)
A magical item (e.g., enchanted sword, spell book)
A landscape (e.g., enchanted forest, mystical mountains)
Each generated image will be combined into a single collage representing a fantasy scene.
"""
import os
import requests
from PIL import Image
from io import BytesIO
import certifi
from openai import OpenAI
from apikey import apikey  # Importing the API key from your apikey.py file

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate a fantasy element
def generate_fantasy_element(prompt):
    """
    Generate an individual fantasy element based on the provided prompt.
    :param prompt: The description of the fantasy element to generate.
    :return: A PIL image of the generated element.
    """
    print(f"Generating fantasy element with prompt: {prompt}")

    # Generate the image using the prompt
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024",  # High-quality image size
        response_format="url"
    )

    image_url = response.data[0].url
    print(f"Generated element: {image_url}")

    # Fetch the image from the URL
    image_response = requests.get(image_url, verify=certifi.where())
    img = Image.open(BytesIO(image_response.content))

    return img


# Function to create a complete fantasy scene by combining elements
def create_fantasy_scene(character, item, landscape):
    """
    Create a complete fantasy scene by combining a character, item, and landscape.
    :param character: The character image.
    :param item: The item image.
    :param landscape: The landscape image.
    :return: The final combined fantasy scene.
    """
    # Create a blank canvas for the scene
    scene_width = 1024
    scene_height = 1024
    scene_image = Image.new('RGB', (scene_width, scene_height), color=(255, 255, 255))

    # Resize images for the scene
    character = character.resize((int(scene_width * 0.4), scene_height), Image.Resampling.LANCZOS)
    item = item.resize((int(scene_width * 0.2), int(scene_height * 0.2)), Image.Resampling.LANCZOS)
    landscape = landscape.resize((scene_width, scene_height), Image.Resampling.LANCZOS)

    # Paste the landscape first
    scene_image.paste(landscape, (0, 0))

    # Position the character
    character_x = int(scene_width * 0.1)
    character_y = 0
    scene_image.paste(character, (character_x, character_y), character)

    # Position the magical item
    item_x = int(scene_width * 0.75)
    item_y = int(scene_height * 0.75)
    scene_image.paste(item, (item_x, item_y), item)

    return scene_image


# Main function to run the fantasy scene generator
def main():
    # Example use cases for different fantasy elements
    character_prompt = "A brave knight in shining armor standing heroically"
    item_prompt = "An enchanted sword glowing with magical energy"
    landscape_prompt = "A mystical forest with glowing trees and magical creatures"

    # Generate each element
    character_image = generate_fantasy_element(character_prompt)
    item_image = generate_fantasy_element(item_prompt)
    landscape_image = generate_fantasy_element(landscape_prompt)

    # Create the final fantasy scene
    final_scene = create_fantasy_scene(character_image, item_image, landscape_image)
    final_scene.show()

    # Save the final scene image
    output_name = "fantasy_scene_image.png"
    final_scene.save(output_name)
    print(f"Fantasy scene image saved as {output_name}")


if __name__ == "__main__":
    main()
"""
Key Learning Points:
Element Generation: You'll learn how to generate different elements (character, item, landscape) separately using AI.
Image Composition: The project focuses on combining multiple images into a single cohesive fantasy scene.
Dynamic Prompts: You'll be able to adjust prompts based on the type of element you want to create, enabling creative freedom.
Example Use Cases:
Character: "A powerful sorceress casting spells"

The program will generate an image of a sorceress in a magical pose.
Magical Item: "A mystical staff adorned with jewels"

The program will create an image of a magical staff with glowing elements.
Landscape: "A serene lake surrounded by enchanted mountains"

The program will depict a beautiful, serene landscape.
Challenge for Further Improvement:
Randomized Element Generation: Create a feature that randomly selects different types of characters, items, and landscapes from predefined lists to enhance variability in scenes.
User Inputs: Allow users to input their own prompts for the character, item, and landscape, giving them more control over the scene.
Additional Elements: Consider adding more elements to the scene, such as additional characters or creatures, to create a richer visual narrative.
This project is designed to elevate your understanding of OpenAI's image generation capabilities while also allowing for creative exploration in fantasy world-building.
"""

"""
remark: still showing a small error
"""