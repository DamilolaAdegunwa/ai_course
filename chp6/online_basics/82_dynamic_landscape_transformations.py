"""
Project Title: Dynamic Landscape Transformations
File Name: dynamic_landscape_transformations.py
Description:
This project involves creating stunning digital landscapes that transition through different times of the day or weather conditions. Using a single prompt, you will generate variations of landscapes reflecting morning, noon, evening, and night scenarios. Each variation will showcase unique lighting, colors, and atmospheric effects, allowing you to explore the capabilities of AI-generated imagery in capturing the essence of nature across different settings.

This project encourages creativity by prompting you to think about how various elements such as time, light, and weather can transform a landscape and how to articulate those transformations into visual concepts.

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


# Function to generate landscape images based on different times of the day
def generate_landscape_variation(prompt, time_of_day):
    """
    Generate a landscape image based on the given time of day.

    :param prompt: The base prompt describing the landscape.
    :param time_of_day: The time of day to focus on (morning, noon, evening, night).
    :return: URL of the generated landscape image.
    """
    full_prompt = f"{prompt} during {time_of_day}"
    response = client.images.generate(
        prompt=full_prompt,
        size="1024x1024"  # High resolution for detailed landscapes
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
    # Define the base prompt for the landscape
    base_landscape_prompt = "A serene mountain landscape with a river flowing through it"
    times_of_day = ["morning", "noon", "evening", "night"]

    # Generate, download, and save each landscape image for different times of day
    for time in times_of_day:
        print(f"Generating image for {time}...")
        image_url = generate_landscape_variation(base_landscape_prompt, time)
        image = download_image(image_url)
        save_image(image, f"landscape_{time}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Base Prompt: "A serene mountain landscape with a river flowing through it"
Time of Day: "morning"
Expected Output:
An image depicting a mountain landscape bathed in soft morning light, with golden hues illuminating the river.
Input:

Base Prompt: "A serene mountain landscape with a river flowing through it"
Time of Day: "noon"
Expected Output:
A bright and vibrant image showing the same landscape under clear blue skies with intense sunlight reflecting off the water.
Input:

Base Prompt: "A serene mountain landscape with a river flowing through it"
Time of Day: "evening"
Expected Output:
A warm, colorful image illustrating a sunset over the mountains, with shades of orange and purple painting the sky.
Input:

Base Prompt: "A serene mountain landscape with a river flowing through it"
Time of Day: "night"
Expected Output:
A serene night scene showcasing a starry sky above the mountains, with the river reflecting moonlight.
Input:

Base Prompt: "A tranquil beach at sunset"
Time of Day: "morning"
Expected Output:
A peaceful morning beach scene with soft light, gentle waves, and a pastel-colored sky as the sun rises.
Project Overview:
In this project, you'll experiment with how the same landscape can transform dramatically based on time of day. Each image generated will highlight the unique qualities of natural light and weather, allowing you to explore various artistic interpretations. This exercise not only enhances your technical skills in image generation but also sharpens your ability to communicate visual ideas effectively.







"""
