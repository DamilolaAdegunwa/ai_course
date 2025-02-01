"""
Project Title:
Theme-Based AI Art Collections

File Name:
theme_based_ai_art_collections.py

Project Description:
This project will create a Python script that generates a themed collection of AI-generated images based on a specific topic. You’ll generate a set of images using a consistent theme, such as a season, cultural elements, or a specific art movement. The theme will influence the style, mood, and content of each image within the collection, giving you the opportunity to work with prompts that require consistency across multiple outputs.

The script will generate several images using a single theme and allow you to test the theme's consistency by producing various images that fit within it.

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
    """Generates an image based on the prompt for a themed art collection."""

    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",  # Modify based on your preference (e.g., 512x512)
        n=1  # Number of images to generate
    )

    return response.data[0].url


def generate_collection(theme, variations):
    """Generates a collection of images based on a theme and variations."""

    print(f"Generating collection for the theme: {theme}")
    image_urls = []

    for i, variation in enumerate(variations):
        full_prompt = f"{theme}, {variation}"
        print(f"Generating image {i + 1} with prompt: {full_prompt}")

        image_url = generate_image_from_prompt(full_prompt)
        image_urls.append(image_url)
        print(f"Image {i + 1} URL: {image_url}")

    return image_urls


# Example usage:
if __name__ == "__main__":
    theme = "Autumn Forest"
    variations = [
        "with soft golden light filtering through the trees",
        "under a cloudy sky, with vibrant red and orange leaves",
        "in the evening, with a misty atmosphere and fallen leaves",
        "with a wooden cabin, surrounded by trees and a stream",
        "during sunset, with dramatic shadows and golden hues"
    ]

    collection_urls = generate_collection(theme, variations)
"""
Example Inputs and Expected Outputs:
Input:
Theme: Autumn Forest
Variation: with soft golden light filtering through the trees
Expected Output:
A URL linking to an image of a peaceful forest during autumn, with sunlight filtering through the yellow-orange leaves of the trees.

Input:
Theme: Autumn Forest
Variation: under a cloudy sky, with vibrant red and orange leaves
Expected Output:
A URL linking to an image showing an autumn forest with a dark, cloudy sky and bright red and orange leaves standing out in the overcast lighting.

Input:
Theme: Autumn Forest
Variation: in the evening, with a misty atmosphere and fallen leaves
Expected Output:
A URL linking to an image depicting a misty forest in the evening with scattered fallen leaves and muted colors.

Input:
Theme: Autumn Forest
Variation: with a wooden cabin, surrounded by trees and a stream
Expected Output:
A URL linking to an image of a cozy wooden cabin nestled in a forest with a small stream running nearby, capturing the warm feeling of autumn.

Input:
Theme: Autumn Forest
Variation: during sunset, with dramatic shadows and golden hues
Expected Output:
A URL linking to an image showing an autumn forest at sunset, with dramatic shadows and glowing golden light illuminating the trees and leaves.

This project focuses on building themed image collections, allowing you to explore how variations on a theme can change the mood or style while maintaining a consistent setting or subject matter. It helps you learn about prompt crafting for consistency across a series of generated images.
"""