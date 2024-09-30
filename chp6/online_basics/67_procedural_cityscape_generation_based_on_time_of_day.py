"""
Project Title:
Procedural Cityscape Generation Based on Time of Day

File Name:
procedural_cityscape_generation_based_on_time_of_day.py

Project Description:
This project focuses on creating a dynamic procedural cityscape generator that produces city landscapes based on a user-defined time of day. The script allows you to specify a prompt for a cityscape with variations like "morning," "afternoon," "sunset," "night," or "dawn." The AI will generate a corresponding cityscape that reflects the time-based ambiance and lighting. This project enhances your skills in combining abstract concepts like time with realistic elements like lighting and scenery to produce diverse outputs.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # The file containing the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_image_from_prompt(prompt):
    """Generates a cityscape image based on the time of day specified in the prompt."""

    # Generate the image using the OpenAI API
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",  # You can change this to "512x512" or other sizes if needed
        n=1  # Number of images to generate
    )

    # Return the URL of the generated image
    return response.data[0].url


def generate_cityscapes(time_of_day_variations):
    """Generates cityscapes based on different times of the day."""
    cityscape_images = []
    for i, time_of_day in enumerate(time_of_day_variations):
        print(f"Generating cityscape for: {time_of_day}")
        image_url = generate_image_from_prompt(f"cityscape at {time_of_day}")
        cityscape_images.append(image_url)
        print(f"Generated image URL for {time_of_day}: {image_url}")
    return cityscape_images


# Example usage:
if __name__ == "__main__":
    # Different time of day themes for cityscapes
    cityscape_times = [
        "early morning with misty streets",
        "afternoon with vibrant sunlight",
        "city skyline at sunset with golden hour lighting",
        "night with neon lights and dark streets",
        "dawn with the first light of the day"
    ]

    # Generate the cityscape images
    cityscape_images = generate_cityscapes(cityscape_times)

    # Print the URLs of the generated images
    for i, img_url in enumerate(cityscape_images):
        print(f"Cityscape Image {i + 1}: {img_url}")
"""
Example Inputs and Expected Outputs:
Input:
Theme: early morning with misty streets
Expected Output:
A URL linking to an image of a quiet city in the early morning, with light fog covering the streets, soft lighting, and peaceful ambiance.

Input:
Theme: afternoon with vibrant sunlight
Expected Output:
A URL linking to an image of a bustling city in broad daylight, with clear skies and bright sunlight reflecting off glass buildings.

Input:
Theme: city skyline at sunset with golden hour lighting
Expected Output:
A URL linking to an image of a city skyline bathed in warm golden hour light, with the sun setting in the background, casting long shadows.

Input:
Theme: night with neon lights and dark streets
Expected Output:
A URL linking to an image of a vibrant city at night, filled with neon signs, dark streets, and glowing windows from high-rise buildings.

Input:
Theme: dawn with the first light of the day
Expected Output:
A URL linking to an image of a quiet city just before sunrise, with soft morning colors in the sky and faint light touching the tops of buildings.

This project adds a time-based variation to image generation, pushing you to think more deeply about how lighting, colors, and ambiance change throughout the day in urban environments.







"""