"""
Project Title:
Dynamic Theme-Based Image Generation

File Name:
dynamic_theme_based_image_generation.py

Project Description:
In this project, you will create a Python script that generates images based on predefined themes, such as "futurism," "nature," or "urban landscapes." The user provides a theme, and the script automatically enriches the theme by adding related concepts and stylistic elements. This project aims to introduce theme-based image generation and enhance creativity by incorporating thematic variations. Additionally, users can explore how different themes affect the artistic output, giving more control and flexibility to the generation process.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # The file containing the API key
import random

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def enrich_theme(theme):
    """Enriches the theme with relevant styles and concepts."""
    themes_dict = {
        "futurism": [
            "neon colors",
            "high-tech cities",
            "futuristic vehicles",
            "digital art style"
        ],
        "nature": [
            "lush forests",
            "sunsets over mountains",
            "serene lakes",
            "wild animals in the wild"
        ],
        "urban": [
            "modern architecture",
            "crowded streets",
            "city skylines",
            "graffiti-covered walls"
        ],
        "fantasy": [
            "dragons flying",
            "ancient ruins",
            "magical forests",
            "mythical creatures"
        ],
        "space": [
            "galaxies in the distance",
            "alien planets",
            "spaceships exploring the unknown",
            "asteroid fields"
        ]
    }

    # Select relevant concepts for the chosen theme
    concepts = random.sample(themes_dict.get(theme, []), 2)
    enriched_theme = f"{theme} with {concepts[0]} and {concepts[1]}"
    return enriched_theme


def generate_image_from_prompt(prompt):
    """Generates an image from a user-defined theme prompt."""
    # Generate the image based on the enriched theme prompt
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",  # Size can be "512x512" or "256x256" depending on the need
        n=1  # Number of images to generate
    )

    # Extract and return the image URL
    return response.data[0].url


def generate_images_for_themes(themes):
    """Generates images for a list of themes."""
    images = []
    for i, theme in enumerate(themes):
        enriched_theme = enrich_theme(theme)
        print(f"Generating image for theme {i + 1}: {enriched_theme}")
        image_url = generate_image_from_prompt(enriched_theme)
        images.append(image_url)
        print(f"Image {i + 1} URL: {image_url}")
    return images


# Example usage:
if __name__ == "__main__":
    # A list of themes to generate images for
    themes = ["futurism", "nature", "urban", "fantasy", "space"]

    # Generate images for each theme
    theme_images = generate_images_for_themes(themes)
    for i, img_url in enumerate(theme_images):
        print(f"Theme {i + 1} Image URL: {img_url}")
"""
Example Inputs and Expected Outputs:
Input:
Theme: futurism
Enriched Theme (example): futurism with neon colors and high-tech cities
Expected Output:
A URL linking to an image of a futuristic city with bright neon colors and advanced technology.

Input:
Theme: nature
Enriched Theme (example): nature with sunsets over mountains and serene lakes
Expected Output:
A URL linking to an image of a natural landscape featuring a calm lake and mountains illuminated by a beautiful sunset.

Input:
Theme: urban
Enriched Theme (example): urban with crowded streets and modern architecture
Expected Output:
A URL linking to an image of a busy cityscape, filled with people and sleek, modern buildings.

Input:
Theme: fantasy
Enriched Theme (example): fantasy with dragons flying and ancient ruins
Expected Output:
A URL linking to an image of a fantasy scene with dragons soaring in the sky and mysterious ancient ruins in the foreground.

Input:
Theme: space
Enriched Theme (example): space with galaxies in the distance and alien planets
Expected Output:
A URL linking to an image of a vast space scene, showcasing distant galaxies and otherworldly planets.

This project takes your understanding of AI image generation to the next level by introducing thematic variations. With pre-defined themes and enriched prompts, users can explore how different themes are interpreted visually by the AI, fostering creativity and experimentation.
"""