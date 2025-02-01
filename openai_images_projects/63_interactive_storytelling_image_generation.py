"""
Project Title:
Interactive Storytelling Through AI-Generated Imagery

File Name:
interactive_storytelling_image_generation.py

Project Description:
In this project, you will create a Python script that dynamically generates a sequence of images based on an evolving narrative or storyline. The script allows users to input multiple stages or scenes from a story, and the AI generates corresponding images to visually represent each stage. This project enhances your skills by combining creative storytelling with image generation. It also encourages you to experiment with multi-step processes where each image contributes to a larger narrative.

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
    """Generates an image from a user-defined story prompt."""

    # Generate the image based on the user-defined story prompt
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",  # You can choose other sizes like "512x512"
        n=1  # Number of images to generate
    )

    # Extract and return the image URL
    return response.data[0].url


def generate_story_images(story_scenes):
    """Generates a series of images based on the narrative scenes of a story."""
    images = []
    for i, scene in enumerate(story_scenes):
        print(f"Generating image for scene {i + 1}: {scene}")
        image_url = generate_image_from_prompt(scene)
        images.append(image_url)
        print(f"Image {i + 1} URL: {image_url}")
    return images


# Example usage:
if __name__ == "__main__":
    # A list of narrative scenes from a story
    story_scenes = [
        "A young warrior standing at the edge of a dark forest, sword in hand, with the setting sun behind him",
        "The warrior enters the forest, where ancient trees loom high and strange creatures watch from the shadows",
        "A fierce battle between the warrior and a giant, with glowing eyes, beneath a stormy sky",
        "The warrior standing victorious on a mountain peak, gazing at the horizon as dawn breaks",
        "A final scene where the warrior returns home to a village, welcomed by cheering crowds"
    ]

    # Generate images for each scene
    story_images = generate_story_images(story_scenes)
    for i, img_url in enumerate(story_images):
        print(f"Story Scene {i + 1}: {img_url}")
"""
Example Inputs and Expected Outputs:
Input:
Scene: A young warrior standing at the edge of a dark forest, sword in hand, with the setting sun behind him
Expected Output:
A URL linking to an image depicting a warrior with a sword, the edge of a dark forest in the background, and the warm glow of the setting sun lighting the scene.

Input:
Scene: The warrior enters the forest, where ancient trees loom high and strange creatures watch from the shadows
Expected Output:
A URL linking to an image showing a dense, mysterious forest with towering trees and hidden creatures lurking in the shadows, with the warrior cautiously stepping forward.

Input:
Scene: A fierce battle between the warrior and a giant, with glowing eyes, beneath a stormy sky
Expected Output:
A URL linking to an image of an intense battle scene, featuring a towering giant with glowing eyes and the warrior, all set under dark, stormy clouds.

Input:
Scene: The warrior standing victorious on a mountain peak, gazing at the horizon as dawn breaks
Expected Output:
A URL linking to an image of the warrior standing triumphantly on a mountain peak, overlooking a beautiful sunrise that signals the dawn of a new day.

Input:
Scene: A final scene where the warrior returns home to a village, welcomed by cheering crowds
Expected Output:
A URL linking to an image of a triumphant homecoming scene, with the warrior being greeted by a crowd of villagers, smiling and cheering.

This project enhances your understanding of working with sequential prompts in storytelling and image generation. You can experiment with different genres, from fantasy to science fiction, and see how changing the narrative alters the generated imagery. Each scene contributes to building a larger, cohesive story through images.










"""