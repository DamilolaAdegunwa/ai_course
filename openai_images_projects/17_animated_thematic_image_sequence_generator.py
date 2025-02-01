"""
Project Title: Animated Thematic Image Sequence Generator
Description:
In this project, you will build a Thematic Image Sequence Generator that takes user input for a theme and generates a sequence of images depicting the evolution or transformation of that theme over time or phases. For example, if the theme is "Tree," the sequence could show a sapling, a young tree, and a fully grown tree. This sequence of images can be compiled into a basic animation (GIF format).

This project is more advanced than the previous one because:

Sequential Image Generation: Instead of generating static images, this project generates a sequence of images representing progression.
GIF Creation: The images will be programmatically combined into a GIF to animate the sequence.
More Dynamic Prompts: The user specifies multiple stages or phases for the theme, allowing for more complex generation.
Key Features:
Sequential Image Generation: For each theme, a series of images that represent different stages of transformation will be generated.
Animation (GIF): The generated images will be compiled into a GIF, creating an animated sequence of images.
Customization: The user can select how many stages or phases to represent in the animation.
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
        size="512x512",  # Fixed size for images in the sequence
        response_format="url"
    )
    image_url = response.data[0].url
    print(f"Generated image for prompt '{prompt}': {image_url}")

    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))


# Function to generate a sequence of images based on a theme and stages
def generate_image_sequence(theme, stages):
    """
    Generate a sequence of images representing the evolution or progression of a theme.
    :param theme: The main theme of the sequence.
    :param stages: A list of stages or phases for the theme.
    :return: A list of generated images.
    """
    images = []
    for stage in stages:
        prompt = f"{theme} in the stage of {stage}"
        image = generate_image_from_prompt(prompt)
        images.append(image)

    return images


# Function to create a GIF from a list of images
def create_gif_from_images(images, gif_name="animation.gif", duration=500):
    """
    Create a GIF from a list of images.
    :param images: A list of PIL images.
    :param gif_name: The output filename of the GIF.
    :param duration: The duration between frames in milliseconds.
    :return: The file path of the created GIF.
    """
    images[0].save(
        gif_name,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0  # Loop indefinitely
    )
    print(f"GIF saved as {gif_name}")
    return gif_name


# Main function to run the image sequence generation and GIF creation
def main():
    theme = input("Enter the theme for your image sequence (e.g., Tree, City): ")
    num_stages = int(input("How many stages do you want to generate? "))

    stages = []
    for i in range(num_stages):
        stage = input(f"Enter description for stage {i + 1} (e.g., sapling, young tree): ")
        stages.append(stage)

    # Generate the sequence of images
    images = generate_image_sequence(theme, stages)

    # Create a GIF from the generated images
    gif_name = f"{theme}_animation.gif"
    create_gif_from_images(images, gif_name=gif_name)

    # Show the GIF
    gif_image = Image.open(gif_name)
    gif_image.show()


if __name__ == "__main__":
    main()

"""
Key Learning Points:
Dynamic Sequential Image Generation: You will dynamically generate images that represent different stages of a theme, building on prompt-based image generation from the previous exercise.
GIF Creation: You will learn how to use Pillow to compile multiple images into a looping GIF, creating an animated sequence.
User-Controlled Progression: The user has control over the number of stages and the description of each stage, making this exercise more customizable.
Explanation:
Sequential Image Generation: The user specifies multiple stages (e.g., "sapling", "young tree", "fully grown tree"). Each stage generates a corresponding image, creating a sense of transformation or progression.
GIF Creation: Once the images are generated, they are compiled into an animated GIF using the Pillow library. This introduces an additional layer of complexity as you're dealing with image sequencing and animation.
Looping Animation: The GIF loops indefinitely, making it a simple but effective way to visualize a theme in transformation.
Example Use Case:
Imagine generating an animated sequence of "City Development" with stages like:

"Empty Land"
"Small Buildings"
"Skyscrapers"
"Modern Metropolis"
Each stage would generate an image, and they would be compiled into a GIF that visualizes the development of a city over time.

Advanced Elements:
Progression Logic: Instead of just user input for each stage, you could introduce logic that auto-generates reasonable "next stages" based on the initial theme.
Transition Effects: Experiment with adding simple transition effects between frames to make the GIF smoother or more visually appealing.
Challenges for Further Development:
Custom Durations for Each Frame: Allow the user to specify how long each image in the sequence should be displayed in the GIF.
Add Captions to Each Frame: Use Pillow’s ImageDraw to add captions or descriptions of each stage on top of the generated images before compiling the GIF.
Advanced Theming: Integrate more complex themes, where stages are auto-suggested based on the main theme (e.g., "Seasons of the Year" could auto-generate spring, summer, autumn, and winter).
This project is a noticeable step up in complexity by introducing the concept of dynamic image sequencing and animation while building on the foundation of prompt-based image generation you learned previously.
"""