"""
Project Title:
Conceptual Image Series with Custom Variations

Project Description:
In this project, you'll build a tool that generates images by combining a base concept with various custom modifiers. The twist here is the ability to specify variations such as color schemes, lighting conditions, or specific details to be applied across multiple generated images. This approach will help create more controlled variations on a single idea, with the added flexibility to input specific features to manipulate.

You'll be using different combinations of modifiers like lighting, mood, and color schemes, allowing you to fine-tune each image's style while staying rooted in the same base concept.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_image_series_with_variations(prompt, variations):
    """
    Generate a series of images based on a core concept and custom variations.

    Parameters:
        prompt (str): The base idea or concept for the images.
        variations (list): A list of specific variations to apply (e.g., color, lighting).

    Returns:
        List of image URLs.
    """
    image_urls = []

    for variation in variations:
        response = client.images.generate(
            prompt=f"{prompt}, {variation}",
            size="1024x1024"
        )
        image_urls.append(response.data[0].url)

    return image_urls


# Example Use Cases

# Example 1: Generate a cityscape with different color schemes and lighting
city_variations = ['in golden sunlight', 'at night with neon lights', 'with pastel colors', 'in monochrome',
                   'at sunrise']
city_images = generate_image_series_with_variations("cityscape", city_variations)
print("Cityscape with Various Lighting and Colors:", city_images)

# Example 2: Generate a fantasy castle with different environments
castle_variations = ['on a snowy mountain', 'in a lush forest', 'in a desert', 'floating in the sky', 'underwater']
castle_images = generate_image_series_with_variations("fantasy castle", castle_variations)
print("Fantasy Castle with Various Environments:", castle_images)

# Example 3: Generate portraits of a character with different moods and styles
character_variations = ['angry expression', 'smiling in warm light', 'in cyberpunk style', 'with dark shadows',
                        'wearing a crown']
character_images = generate_image_series_with_variations("portrait of a warrior", character_variations)
print("Character Portraits with Different Moods and Styles:", character_images)

# Example 4: Generate forest scenes with different atmospheres
forest_variations = ['in dense fog', 'in autumn colors', 'at night with stars', 'with magical creatures',
                     'in winter snow']
forest_images = generate_image_series_with_variations("forest landscape", forest_variations)
print("Forest Landscape with Various Atmospheres:", forest_images)

# Example 5: Generate spacecraft designs with different futuristic styles
spacecraft_variations = ['sleek and minimalist', 'with steampunk design', 'covered in neon lights',
                         'made of organic material', 'in industrial style']
spacecraft_images = generate_image_series_with_variations("spacecraft", spacecraft_variations)
print("Spacecraft with Various Futuristic Designs:", spacecraft_images)
"""
Example Inputs and Expected Outputs:
Input:

Prompt: "cityscape"
Variations: ['in golden sunlight', 'at night with neon lights', 'with pastel colors', 'in monochrome', 'at sunrise']
Expected Output:

A list of 5 URLs, each showing a cityscape with a different lighting condition or color scheme.
Input:

Prompt: "fantasy castle"
Variations: ['on a snowy mountain', 'in a lush forest', 'in a desert', 'floating in the sky', 'underwater']
Expected Output:

A list of 5 images showcasing different environments for a fantasy castle.
Input:

Prompt: "portrait of a warrior"
Variations: ['angry expression', 'smiling in warm light', 'in cyberpunk style', 'with dark shadows', 'wearing a crown']
Expected Output:

A set of 5 portraits of a warrior, each with a different mood or stylistic choice.
Input:

Prompt: "forest landscape"
Variations: ['in dense fog', 'in autumn colors', 'at night with stars', 'with magical creatures', 'in winter snow']
Expected Output:

A list of 5 URLs displaying different forest landscapes with unique atmospheres.
Input:

Prompt: "spacecraft"
Variations: ['sleek and minimalist', 'with steampunk design', 'covered in neon lights', 'made of organic material', 'in industrial style']
Expected Output:

5 URLs showing various designs of futuristic spacecraft based on the described styles.
Project Summary:
This project expands your ability to explore different visual variations of a concept by allowing you to input specific details for each generated image. It's an excellent way to create themed collections or experiment with different artistic directions within a single subject.
"""