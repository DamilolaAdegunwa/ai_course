"""
Project Title:
Advanced Generative Art Based on Complex Prompts

File Name:
advanced_generative_art.py

Project Description:
In this project, you will create a Python script that generates advanced AI-generated images based on multi-part, complex prompts. The prompts will include attributes like mood, color palette, and artistic style, which can influence the final image. By combining these attributes, you'll be able to generate more dynamic and specific art outputs. This project will focus solely on image generation using detailed textual descriptions.

The script will generate an image based on a user's complex description using OpenAI’s image generation API. You will develop a function that constructs the prompt from multiple elements like artistic styles, emotions, and specific visual elements (e.g., colors, lighting).

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
    """Generates an image from a complex user prompt."""

    # Generate the image based on the complex prompt
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",  # Modify based on your preference (512x512, etc.)
        n=1  # Number of images to generate
    )

    # Extract and return the image URL
    return response.data[0].url


# Example usage:
if __name__ == "__main__":
    prompts = [
        "A serene mountain landscape at sunrise with soft pastel colors and Impressionist style",
        "A futuristic cityscape at night with neon lights, inspired by cyberpunk art",
        "A portrait of a medieval knight in full armor, painted in a realistic Renaissance style",
        "A minimalistic black-and-white geometric design, focused on symmetry and balance",
        "An underwater scene with coral reefs, vibrant marine life, and sunlight filtering through the water, in the style of surrealism"
    ]

    for i, prompt in enumerate(prompts):
        print(f"Generating image {i + 1}...")
        image_url = generate_image_from_prompt(prompt)
        print(f"Image {i + 1} URL: {image_url}")
"""
Example Inputs and Expected Outputs:
Input:
A serene mountain landscape at sunrise with soft pastel colors and Impressionist style Expected Output:
A URL linking to an image that depicts a calm mountain scene with soft, blended colors and a brushstroke style similar to Impressionism.

Input:
A futuristic cityscape at night with neon lights, inspired by cyberpunk art Expected Output:
A URL linking to an image of a glowing, advanced city skyline under a dark sky, illuminated with neon light effects.

Input:
A portrait of a medieval knight in full armor, painted in a realistic Renaissance style Expected Output:
A URL linking to an image of a knight wearing shining armor, rendered in the highly detailed style typical of Renaissance portraits.

Input:
A minimalistic black-and-white geometric design, focused on symmetry and balance Expected Output:
A URL linking to a simple, abstract image using sharp lines, geometric shapes, and contrasts in black and white to emphasize symmetry.

Input:
An underwater scene with coral reefs, vibrant marine life, and sunlight filtering through the water, in the style of surrealism Expected Output:
A URL linking to a dreamlike underwater world with exaggerated features, bright colors, and a touch of surreal artistic distortion.

This project will challenge your ability to write more specific and varied prompts, while also offering insight into how text influences visual outputs when working with AI. The multiple prompt styles (ranging from abstract designs to realistic scenes) will give you hands-on experience with a wide range of creative possibilities!







"""