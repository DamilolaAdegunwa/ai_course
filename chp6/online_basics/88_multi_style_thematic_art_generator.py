"""
Project Title: Multi-Style Thematic Art Generator
File Name: multi_style_thematic_art_generator.py

Project Description:
This project generates a variety of artistic images based on a given theme, applying different artistic styles to each image. The user can specify a theme, and the system will return images rendered in several popular styles (e.g., realism, surrealism, abstract, and minimalism). The project aims to explore the same subject through different artistic lenses, allowing for an in-depth exploration of visual themes.

This exercise introduces multi-style prompts in a single request and uses specific combinations of artistic styles and image concepts. The code dynamically generates prompts and ensures a robust understanding of how AI interprets artistic variations of the same subject.

Python Code:
python
"""
from openai import OpenAI
from apikey import apikey  # apikey.py file contains the OpenAI key

import os

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Define function to generate thematic art in multiple styles
def generate_thematic_art(theme):
    styles = ["realism", "surrealism", "abstract", "minimalism", "impressionism"]
    generated_images = []

    for style in styles:
        prompt = f"A {style} art of {theme} with intricate details and vibrant colors"
        response = client.images.generate(
            prompt=prompt,
            size="1024x1024"
        )
        image_url = response.data[0].url
        generated_images.append((style, image_url))

    return generated_images


# Example usage
if __name__ == "__main__":
    theme = "a peaceful forest"
    images = generate_thematic_art(theme)

    for style, url in images:
        print(f"Style: {style} - Image URL: {url}")
"""
Example Inputs and Expected Outputs:
Input: theme = "a peaceful forest" Expected Output:

Realism: Image URL with a realistic forest
Surrealism: Image URL with a surreal version of a forest (e.g., floating trees, unusual colors)
Abstract: Image URL with abstract representations of forest elements
Minimalism: Image URL with a simple, minimal representation of a forest
Impressionism: Image URL in an impressionist style, with soft brush strokes and light effects
Input: theme = "a futuristic city" Expected Output:

Realism: Image URL with realistic tall skyscrapers, advanced technologies
Surrealism: Image URL with dreamlike, distorted cityscapes
Abstract: Image URL with abstract lines, shapes representing the city
Minimalism: Image URL showing a sleek, minimal futuristic city
Impressionism: Image URL showing a city with bright lights and flowing strokes
Input: theme = "an ancient temple" Expected Output:

Realism: Image URL with historically accurate temple details
Surrealism: Image URL with a mystical version of the temple, blending reality and imagination
Abstract: Image URL with temple shapes represented through abstract geometry
Minimalism: Image URL with clean lines and few temple details
Impressionism: Image URL with soft lighting effects and an atmospheric temple
Input: theme = "a mountain range" Expected Output:

Realism: Image URL with realistic mountain scenery
Surrealism: Image URL with exaggerated mountain shapes, floating peaks
Abstract: Image URL with geometric and color-based representations of mountains
Minimalism: Image URL with a few essential lines and shapes depicting mountains
Impressionism: Image URL with soft, glowing lighting of a sunset over mountains
Input: theme = "a bustling market" Expected Output:

Realism: Image URL with detailed scenes of a crowded, busy market
Surrealism: Image URL with strange and whimsical market scenes
Abstract: Image URL with abstract colors and shapes representing the market
Minimalism: Image URL with simple, clear lines showing only the essential market elements
Impressionism: Image URL with vibrant colors and loose strokes of market activity
This project offers variety, exploring how different artistic styles interpret the same theme. The approach involves both generating and comparing different perspectives, resulting in a comprehensive visual exploration of any concept.







"""