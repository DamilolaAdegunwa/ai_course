"""
Project Title:
Multi-Style Image Generation Tool

Project Description:
In this advanced project, you'll build a tool that generates images from the same concept but in varying styles. Instead of producing a simple set of variations, this project allows you to control the artistic style of each image based on a list of predefined styles. This will enhance your ability to explore how different artistic genres or moods can transform a concept.

The core idea is to generate different representations of a single prompt, such as "ancient city," and apply distinct styles like watercolor, cyberpunk, minimalism, or abstract art. You can use this project to generate diverse visual interpretations of the same subject and study how various stylistic choices affect the result.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_images_with_styles(prompt, styles):
    """
    Generate images based on a single prompt and multiple styles.

    Parameters:
        prompt (str): The concept or subject for the image generation.
        styles (list): A list of artistic styles or descriptors for the images.

    Returns:
        List of image URLs.
    """
    image_urls = []

    # Iterate over the styles to generate different artistic interpretations
    for style in styles:
        response = client.images.generate(
            prompt=f"{prompt} in {style} style",
            size="1024x1024"
        )
        image_urls.append(response.data[0].url)

    return image_urls


# Example Use Cases

# Example 1: Generate an ancient city in different styles
city_styles = ['watercolor', 'cyberpunk', 'minimalism', 'abstract', 'realistic']
city_images = generate_images_with_styles("ancient city", city_styles)
print("Ancient City in Various Styles:", city_images)

# Example 2: Generate space stations with multiple artistic interpretations
space_styles = ['pixel art', 'fantasy', 'steampunk', 'anime', '3D render']
space_images = generate_images_with_styles("space station", space_styles)
print("Space Station in Various Styles:", space_images)

# Example 3: Generate different versions of a forest landscape
forest_styles = ['impressionism', 'dark fantasy', 'pop art', 'futuristic', 'surrealism']
forest_images = generate_images_with_styles("forest landscape", forest_styles)
print("Forest Landscape in Various Styles:", forest_images)

# Example 4: Generate a set of dragon concepts in various themes
dragon_styles = ['cartoon', 'gothic', 'low-poly', 'ink drawing', 'neon art']
dragon_images = generate_images_with_styles("dragon", dragon_styles)
print("Dragon in Various Styles:", dragon_images)

# Example 5: Generate a futuristic city with contrasting artistic styles
futuristic_styles = ['sci-fi', 'post-apocalyptic', 'utopian', 'digital painting', 'oil painting']
futuristic_images = generate_images_with_styles("futuristic city", futuristic_styles)
print("Futuristic City in Various Styles:", futuristic_images)
"""
Example Inputs and Expected Outputs:
Input:

Prompt: "ancient city"
Styles: ['watercolor', 'cyberpunk', 'minimalism', 'abstract', 'realistic']
Expected Output:

A list of 5 URLs, each representing the "ancient city" in different artistic styles (e.g., watercolor, cyberpunk).
Input:

Prompt: "space station"
Styles: ['pixel art', 'fantasy', 'steampunk', 'anime', '3D render']
Expected Output:

A list of 5 URLs, each representing the "space station" interpreted in various artistic genres.
Input:

Prompt: "forest landscape"
Styles: ['impressionism', 'dark fantasy', 'pop art', 'futuristic', 'surrealism']
Expected Output:

A set of 5 images showing a forest landscape in each listed style.
Input:

Prompt: "dragon"
Styles: ['cartoon', 'gothic', 'low-poly', 'ink drawing', 'neon art']
Expected Output:

A list of 5 URLs showing different dragon designs based on the provided styles.
Input:

Prompt: "futuristic city"
Styles: ['sci-fi', 'post-apocalyptic', 'utopian', 'digital painting', 'oil painting']
Expected Output:

5 URLs showing various representations of a futuristic city in different styles.
Project Summary:
This project helps you master generating images in different artistic styles, using the same prompt. You'll learn to control style-based variations, making it ideal for generating multiple artistic interpretations of a concept. It can be applied to a wide variety of use cases such as game design, concept art, and moodboards.
"""