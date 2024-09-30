"""
Project Title:
AI-Generated Mood Boards Based on Custom Concepts

File Name:
ai_generated_mood_boards.py

Project Description:
In this project, you'll create a Python script that generates a "mood board" of images based on different user-defined concepts. The script allows users to input various ideas or emotions (such as "calm," "adventure," or "mystery"), and the AI will generate multiple images that collectively represent the desired mood. This project is advanced in the sense that it encourages creativity by combining multiple abstract ideas and translating them into visual representations.

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
    """Generates an image from a user-defined concept."""

    # Generate the image based on the user-defined concept
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",  # Adjust size as needed (e.g., "512x512")
        n=1  # Number of images to generate per concept
    )

    # Extract and return the image URL
    return response.data[0].url


def generate_mood_board(concepts):
    """Generates a mood board based on a list of user-defined concepts."""
    mood_board_images = []
    for i, concept in enumerate(concepts):
        print(f"Generating image for concept {i + 1}: {concept}")
        image_url = generate_image_from_prompt(concept)
        mood_board_images.append(image_url)
        print(f"Image {i + 1} URL: {image_url}")
    return mood_board_images


# Example usage:
if __name__ == "__main__":
    # A list of user-defined concepts for the mood board
    mood_board_concepts = [
        "calm ocean waves at sunset",
        "mystical forest with glowing mushrooms",
        "city at night with neon lights",
        "a mountain view with rolling clouds",
        "a cozy cabin with snow falling outside"
    ]

    # Generate the mood board images
    mood_board_images = generate_mood_board(mood_board_concepts)

    # Print out the URLs of the generated images
    for i, img_url in enumerate(mood_board_images):
        print(f"Mood Board Image {i + 1}: {img_url}")
"""
Example Inputs and Expected Outputs:
Input:
Concept: calm ocean waves at sunset
Expected Output:
A URL linking to an image of a serene beach scene with gentle waves rolling in at sunset.

Input:
Concept: mystical forest with glowing mushrooms
Expected Output:
A URL linking to an image of a dark, enchanted forest illuminated by otherworldly glowing mushrooms.

Input:
Concept: city at night with neon lights
Expected Output:
A URL linking to an image of an urban cityscape at night, featuring bright neon lights and bustling streets.

Input:
Concept: a mountain view with rolling clouds
Expected Output:
A URL linking to an image of a panoramic mountain view with thick clouds swirling around the peaks.

Input:
Concept: a cozy cabin with snow falling outside
Expected Output:
A URL linking to an image of a warm, wooden cabin nestled in a snowy landscape, with gentle snowflakes falling.

This project pushes your ability to creatively combine concepts and generate multiple images that collectively reflect a mood or theme. By generating a series of images, you can create cohesive mood boards for various use cases, such as design, presentations, or storytelling.







"""