# 15 https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833
"""
Project Title: AI-Generated Interactive Storyboard
Description:
This exercise will focus on generating a series of images for an interactive storyboard based on user input prompts. The prompts describe different scenes, and you will generate an image for each scene to visually narrate a short story. You’ll also learn to display these images in a sequence, mimicking the visual structure of a comic or a storyboard.

The project is noticeably more advanced because it integrates user input to dynamically build a coherent story, progresses through multiple scenes, and displays the sequence of generated images as a unified storyboard. You'll also work with multiple AI-generated prompts at once and organize the output efficiently.

Key Features:
Dynamic Storyboarding: Users provide a series of scene prompts for a short narrative, and AI generates corresponding images for each scene.
Image Sequence Display: The images are organized and displayed as a storyboard, simulating a comic strip or a visual outline of the story.
Custom Story Flow: The project dynamically handles different user-provided prompts and ties them together visually.
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
    Generate an image based on a text prompt using OpenAI's DALL·E 3 model.
    """
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024",  # Static size for all images
        response_format="url"
    )
    image_url = response.data[0].url
    print(f"Generated image for prompt '{prompt}': {image_url}")

    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))


# Function to generate a storyboard based on a sequence of prompts
def generate_storyboard(scene_prompts):
    """
    Generate a storyboard by generating images for each scene prompt.
    :param scene_prompts: A list of scene prompts to generate images for.
    :return: A list of generated PIL images.
    """
    images = []
    for i, prompt in enumerate(scene_prompts):
        print(f"Generating image for scene {i + 1}...")
        image = generate_image_from_prompt(prompt)
        images.append(image)
    return images


# Function to display storyboard images in sequence
def display_storyboard(images):
    """
    Display all images in the storyboard.
    :param images: A list of PIL images to display as a storyboard.
    """
    for i, image in enumerate(images):
        image.show(title=f"Scene {i + 1}")


# Main function to run the interactive storyboard generation
def main():
    # Ask the user to input scene descriptions for their story
    scene_count = int(input("How many scenes do you want to generate for your storyboard? "))
    scene_prompts = []

    for i in range(scene_count):
        prompt = input(f"Enter the description for scene {i + 1}: ")
        scene_prompts.append(prompt)

    # Generate images for all scene prompts
    storyboard_images = generate_storyboard(scene_prompts)

    # Display the storyboard images in sequence
    display_storyboard(storyboard_images)


if __name__ == "__main__":
    main()

"""
Key Learning Points:
User-Driven Image Generation: You’ll generate multiple images based on user-defined text prompts, which adds a layer of dynamic input processing.
Storyboard Construction: You will not only generate images but also organize them sequentially to create a visual story.
Multiple API Calls: Managing multiple API calls to OpenAI’s image generator within a loop requires you to handle responses effectively.
Interactive Input Handling: The program takes input from the user to drive the entire process, giving the user control over the story's direction.
Explanation:
Dynamic Input: The user can specify multiple scene descriptions, which creates flexibility in the types of images generated. Each prompt leads to a unique image.
Sequential Generation: The images are generated and stored in a list, allowing the user to see the flow of their visual story.
Fixed Size: The images are generated at a static size of 1024x1024, as per the exercise requirements.
Image Display: The images are displayed sequentially using PIL.Image.show() to simulate the experience of flipping through a storyboard or comic.
Advanced Elements:
Sequential Input and Output: This is an important step towards building more complex projects where the user defines input sequences that dynamically change the outcome.
User Interaction: Adding input prompts and multiple scene handling pushes your understanding of integrating user-driven workflows with AI.
Additional Challenges:
Story Continuity: Try modifying the scene prompts between frames based on the previous scene to ensure continuity and narrative progression.
Image Stitching: Combine all the generated images into a single composite image, where each scene is placed side by side, resembling a comic strip.
Animation Extension: You could extend this project into an animated format by creating multiple frames for each scene, progressively modifying prompts for smoother transitions.
This exercise will advance your understanding of how to create dynamic, user-driven image generation projects. It builds on the concept of handling multiple image generations, sequential inputs, and displaying visual output in a structured way, which is key for building more interactive AI-driven visual tools.
"""