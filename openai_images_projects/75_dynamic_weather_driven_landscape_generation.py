"""
Project Title: Dynamic Weather-Driven Landscape Generation
File Name: dynamic_weather_driven_landscape_generation.py
Description:
In this project, you will create stunning AI-generated landscape images based on dynamic weather conditions. The prompt generation will change depending on user inputs related to specific weather conditions such as snow, rain, sunshine, and even more extreme phenomena like thunderstorms or tornadoes. This exercise is designed to develop your ability to craft prompts that blend the weather with natural scenery, enhancing your experience in generating diverse environmental visuals.

The goal is to create landscapes that reflect both the beauty and power of nature under different weather conditions. You’ll explore advanced prompt composition while mastering the interplay between weather patterns and the natural environment.

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


# Function to generate images based on weather conditions
def generate_weather_landscape_image(prompt):
    """
    Generate an image based on the weather-related prompt for dynamic landscapes.

    :param prompt: The prompt describing the landscape and the weather condition.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # Larger size to capture detailed landscapes
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
    # Define multiple prompts for different weather-driven landscapes
    weather_prompts = [
        "A peaceful mountain valley under heavy snowfall with evergreen trees covered in snow and a calm river flowing through",
        "A serene beach landscape during sunset, with light rain creating ripples on the water and dark clouds forming above",
        "A vast desert landscape during a thunderstorm, with lightning striking the sand dunes and strong winds whipping up sand clouds",
        "A dense forest in the middle of autumn with golden leaves falling, illuminated by bright sunlight breaking through the clouds",
        "A rural village surrounded by open fields during a misty morning with dew on the grass and light fog covering the area"
    ]

    # Generate, download, and save each weather-driven landscape image
    for i, prompt in enumerate(weather_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_weather_landscape_image(prompt)
        image = download_image(image_url)
        save_image(image, f"weather_landscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A peaceful mountain valley under heavy snowfall with evergreen trees covered in snow and a calm river flowing through"
Expected Output:
A serene mountain valley blanketed in thick snow, with evergreen trees covered in white. A small river flows calmly through the valley, contrasting with the quiet, cold atmosphere of the snowy landscape.
Input:

Prompt: "A serene beach landscape during sunset, with light rain creating ripples on the water and dark clouds forming above"
Expected Output:
A tranquil beach scene at sunset, with gentle waves and light rain forming small ripples on the water. Dark clouds loom above, adding contrast to the orange and pink hues of the sky.
Input:

Prompt: "A vast desert landscape during a thunderstorm, with lightning striking the sand dunes and strong winds whipping up sand clouds"
Expected Output:
A dramatic desert landscape, with lightning striking the towering sand dunes. The strong winds stir up clouds of sand, creating a powerful and chaotic scene under dark, stormy skies.
Input:

Prompt: "A dense forest in the middle of autumn with golden leaves falling, illuminated by bright sunlight breaking through the clouds"
Expected Output:
A vibrant forest in the height of autumn, with golden and orange leaves slowly falling from the trees. Bright sunlight breaks through scattered clouds, casting warm light over the forest floor.
Input:

Prompt: "A rural village surrounded by open fields during a misty morning with dew on the grass and light fog covering the area"
Expected Output:
A peaceful rural village scene with fog rolling over the open fields. The misty morning atmosphere gives the image a soft, tranquil quality, with dew glistening on the grass and light fog adding mystery to the village landscape.
Project Overview:
This project allows you to practice generating visually rich and diverse landscapes based on specific weather conditions. You will explore how to integrate environmental factors such as snow, rain, fog, and sunlight into your prompts to create dynamic, realistic, or surreal scenes. This project builds on previous exercises by focusing more on the interaction between different elements of the environment, enhancing your skills in advanced prompt crafting for AI image generation.







"""