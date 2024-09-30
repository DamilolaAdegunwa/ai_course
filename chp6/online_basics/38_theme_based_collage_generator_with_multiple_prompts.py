"""
Project Title: Theme-Based Collage Generator with Multiple Prompts
In this advanced project, you will generate a collage of images based on multiple prompts provided by the user. Each prompt will generate an individual image, and the final output will stitch these images together into a collage. This adds complexity by requiring the program to handle multiple API requests, download and manipulate multiple images, and create a final composite image from these parts.

Python Code
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


def generate_image_from_prompt(prompt):
    """
    Generate an image based on a description prompt.

    :param prompt: A string containing the image description.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="512x512"
    )
    return response.data[0].url  # Returns the URL of the generated image


def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: The URL of the image.
    :return: PIL Image object
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def create_collage(image_list, output_path="collage.jpg"):
    """
    Create a collage from a list of PIL images.

    :param image_list: A list of PIL Image objects.
    :param output_path: File path where the collage will be saved.
    :return: None
    """
    # Determine the size of the collage
    collage_width = image_list[0].width * 2  # Assuming a 2x2 grid for simplicity
    collage_height = image_list[0].height * 2

    # Create a blank canvas for the collage
    collage_image = Image.new("RGB", (collage_width, collage_height))

    # Paste each image into the correct position
    collage_image.paste(image_list[0], (0, 0))
    collage_image.paste(image_list[1], (image_list[0].width, 0))
    collage_image.paste(image_list[2], (0, image_list[0].height))
    collage_image.paste(image_list[3], (image_list[0].width, image_list[0].height))

    # Save the final collage image
    collage_image.save(output_path)
    print(f"Collage saved as {output_path}")


def generate_and_create_collage(prompts):
    """
    Generate images from prompts and create a collage.

    :param prompts: A list of descriptive prompts for generating images.
    :return: None
    """
    image_list = []
    for prompt in prompts:
        image_url = generate_image_from_prompt(prompt)
        image = download_image(image_url)
        image_list.append(image)

    # Ensure we have enough images to create a collage (4 images in this example)
    if len(image_list) >= 4:
        create_collage(image_list[:4])  # Create a collage from the first 4 images
    else:
        print("Not enough images to create a collage. Provide at least 4 prompts.")


# Example use cases
if __name__ == "__main__":
    # List of prompts for generating a collage
    prompts = [
        "A surreal landscape with floating islands",
        "A futuristic robot standing in a city",
        "A beautiful mountain range at sunrise",
        "A mysterious forest with glowing trees"
    ]

    # Generate and create a collage of the generated images
    generate_and_create_collage(prompts)
"""
Multiple Example Inputs and Expected Outputs
Input:

Prompt: "A surreal landscape with floating islands"
Expected Output:
Image of floating islands in a dreamlike, surreal setting. File saved in the collage.
Input:

Prompt: "A futuristic robot standing in a city"
Expected Output:
Image of a robot in a futuristic urban environment. File saved in the collage.
Input:

Prompt: "A beautiful mountain range at sunrise"
Expected Output:
Image of a picturesque mountain range with warm, sunrise colors. File saved in the collage.
Input:

Prompt: "A mysterious forest with glowing trees"
Expected Output:
A haunting, magical forest with bioluminescent trees. File saved in the collage.
Input:

Prompt: "A serene ocean view at sunset"
Expected Output:
Image of a tranquil ocean with colorful reflections from the setting sun. If chosen, the file is used in the collage if there are fewer than 4 prompts.
Project Overview
This project introduces complexity by dynamically generating images from multiple prompts and combining them into a 2x2 image collage. It requires working with image downloading, handling multiple API calls, and manipulating images through the Python Imaging Library (PIL). The output is a visually rich collage of images, each representing a different theme or concept based on the user’s prompts.

You’ll gain experience in handling multiple inputs, downloading and processing images, and combining these into a cohesive output. This project also encourages you to experiment with various themes to explore how OpenAI generates different image styles and subjects.
"""