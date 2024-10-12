"""
Project Title: Multi-Prompt Style Fusion Image Generator
Description:
In this project, you will create a Multi-Prompt Style Fusion Image Generator that takes multiple prompts and combines them into a single image. Each prompt represents a different theme or style, and the AI will generate an image that merges these diverse themes or artistic styles together.

This is an advanced project because:

Multiple Prompts: Instead of generating a single image from one prompt, you will generate multiple images from different prompts and then fuse them into a single composite image.
Style Fusion: You can experiment with mixing drastically different styles or themes (e.g., "cyberpunk city" and "medieval castle").
Compositing: The images are combined in an advanced way using blending techniques, adding complexity.
Key Features:
Multi-Prompt Fusion: Users can input several themes or styles, and each prompt generates an image, which is then fused into a composite image.
Advanced Blending: The images are blended together with custom opacity settings for each prompt, creating a unique composite image.
Customization: The user can specify how much influence each prompt has on the final image, giving control over the blending.
"""

import os
import requests
from PIL import Image
from io import BytesIO
import certifi
from openai import OpenAI
from apikey import apikey  # Your apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Function to generate an image from a prompt
def generate_image_from_prompt(prompt):
    """
    Generate an image based on a text prompt using OpenAI's DALL·E model.
    """
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024",  # Fixed size for images in the fusion process
        response_format="url"
    )
    image_url = response.data[0].url
    print(f"Generated image for prompt '{prompt}': {image_url}")

    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))

# Function to blend two images with a given blend ratio
def blend_images(image1, image2, alpha=0.5):
    """
    Blend two images together with the given blend ratio (alpha).
    :param image1: The first image.
    :param image2: The second image.
    :param alpha: Blend ratio, where 0.0 is fully the first image, and 1.0 is fully the second image.
    :return: The blended image.
    """
    return Image.blend(image1, image2, alpha)

# Function to generate a composite image from multiple prompts
def generate_composite_image(prompts, blend_ratios):
    """
    Generate a composite image by blending images generated from multiple prompts.
    :param prompts: A list of text prompts.
    :param blend_ratios: A list of blend ratios corresponding to each prompt.
    :return: The final composite image.
    """
    if len(prompts) < 2:
        raise ValueError("You need at least two prompts to blend images.")
    if len(prompts) != len(blend_ratios):
        raise ValueError("Number of prompts and blend ratios must be the same.")

    # Generate the first image
    composite_image = generate_image_from_prompt(prompts[0])

    # Blend successive images
    for i in range(1, len(prompts)):
        next_image = generate_image_from_prompt(prompts[i])
        composite_image = blend_images(composite_image, next_image, alpha=blend_ratios[i])

    return composite_image

# Main function to run the composite image generation
def main():
    num_prompts = int(input("Enter the number of prompts you want to blend (at least 2): "))

    prompts = []
    blend_ratios = []

    for i in range(num_prompts):
        prompt = input(f"Enter prompt {i + 1}: ")
        prompts.append(prompt)
        if i > 0:  # The first prompt doesn't need a blend ratio
            blend_ratio = float(input(f"Enter blend ratio for prompt {i + 1} (0.0 to 1.0): "))
            blend_ratios.append(blend_ratio)

    # The first blend ratio is 0.0 by default (as it is the base image)
    blend_ratios.insert(0, 0.0)

    # Generate and display the composite image
    composite_image = generate_composite_image(prompts, blend_ratios)
    composite_image.show()

    # Save the final image
    output_name = "composite_image.png"
    composite_image.save(output_name)
    print(f"Composite image saved as {output_name}")

if __name__ == "__main__":
    main()

"""
Key Learning Points:
Multi-Prompt Image Generation: You will dynamically generate images from multiple prompts and blend them into a single composite image.
Blending and Alpha Compositing: You will learn how to use alpha compositing techniques to blend different images together.
User-Controlled Ratios: You can control the amount of influence each prompt has on the final image using blend ratios, adding complexity and customization.
Explanation:
Multi-Prompt Input: Users input multiple prompts representing different themes or artistic styles (e.g., "futuristic city" and "forest"), and the AI generates images for each.
Blending: The images are blended together using an alpha compositing technique, where each prompt can have a different weight in the final image based on the blend ratio.
Dynamic Fusion: The project allows users to control how much each prompt influences the final image, creating more advanced and artistic image fusions.
Example Use Case:
Imagine creating a composite image of "cyberpunk city" and "medieval castle" with blend ratios of 0.7 and 0.3, respectively. The final image could have a unique visual style where futuristic and medieval elements are merged together in a seamless way.

Challenges for Further Development:
Weighted Influence: Allow the user to input more complex blending techniques where one prompt gradually fades into another (e.g., gradient blending).
Multiple Images per Prompt: Experiment with generating multiple images for each prompt and blending them in different combinations.
Randomized Prompts: Add an option to generate a random blend by using a random prompt list and random blend ratios.
This project builds on the previous exercises by introducing the concepts of multiple prompt-based image generation and advanced image blending, adding layers of complexity while remaining accessible.
"""