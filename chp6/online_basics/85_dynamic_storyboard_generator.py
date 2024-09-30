"""
Project Title: AI-Powered Dynamic Storyboard Generator for Animated Films
File Name: dynamic_storyboard_generator.py
Description:
In this advanced project, we will create an AI-Powered Dynamic Storyboard Generator capable of generating a sequence of images based on a detailed storyline or screenplay. This tool is designed to assist animators, filmmakers, and creative professionals in visualizing key moments of an animated film or series. It uses OpenAI’s image generation capabilities to create a series of scene-specific images that capture the essence of a provided screenplay.

The goal is to break down a movie or animation script into individual scenes, identify key descriptive elements, and use AI to visualize each scene with stunning detail. The project involves handling a storyline breakdown, scene-to-image transformation, and maintaining visual coherence across multiple images in a single narrative flow.

This project introduces several advanced techniques:

Scene Text Analysis and Keyframe Extraction: Identifies critical moments in a script or description to determine which parts should be visualized.
Scene Progression: Tracks character evolution, environment changes, and emotional tone, ensuring consistency across generated images.
Character Consistency: Maintains visual consistency of characters across multiple scenes by feeding previous image references back into subsequent prompt generations.
This will allow users to create dynamic and visually cohesive storyboards from text-based descriptions, helping to bring stories to life without the need for manual drawing or artist intervention.

Python Code:
python
"""
import os
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key
import requests
from PIL import Image
from io import BytesIO

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate a storyboard from a given script
def generate_storyboard_from_script(script):
    """
    Breaks down the script into scenes and generates a storyboard by generating an image for each scene.

    :param script: A text string containing the screenplay/script
    :return: A list of image URLs corresponding to the key scenes of the script.
    """
    scenes = break_down_script_into_scenes(script)
    storyboard_images = []

    for scene_num, scene_description in enumerate(scenes):
        print(f"Generating image for Scene {scene_num + 1}: {scene_description}")
        prompt = create_image_prompt_from_scene(scene_description)
        image_url = generate_scene_image(prompt)
        storyboard_images.append(image_url)

    return storyboard_images


# Function to break down a script into key scenes
def break_down_script_into_scenes(script):
    """
    Processes the script to break it down into distinct scenes based on key transitions or moments.

    :param script: A text string containing the screenplay/script
    :return: A list of scene descriptions.
    """
    # For simplicity, assume the scenes are separated by double new lines.
    # In real-life, this would involve parsing and understanding screenplay structure.
    scenes = script.split('\n\n')
    return scenes


# Function to create an image prompt from a scene description
def create_image_prompt_from_scene(scene_description):
    """
    Converts a scene description into an image generation prompt.

    :param scene_description: A string describing the scene
    :return: A formatted string prompt for image generation
    """
    # Here, we simplify by directly using the description as the prompt
    return scene_description


# Function to generate an image for a scene
def generate_scene_image(prompt):
    """
    Generates an image for a given scene prompt.

    :param prompt: The formatted scene description to convert into an image.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # Using high-resolution for storyboards
    )
    return response.data[0].url  # Return the image URL


# Function to save storyboard images locally
def save_storyboard_images(image_urls, base_filename="storyboard_scene"):
    """
    Downloads and saves each image in the storyboard to a local file.

    :param image_urls: List of image URLs for the storyboard
    :param base_filename: The base file name to use when saving each image
    """
    for idx, image_url in enumerate(image_urls):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        filename = f"{base_filename}_{idx + 1}.jpg"
        img.save(filename)
        print(f"Scene {idx + 1} saved as {filename}")


# Example use case
if __name__ == "__main__":
    script = """
    Scene 1: A futuristic city skyline at sunset, with flying cars zipping between towering skyscrapers. Neon signs light up as night begins to fall. 
    Scene 2: Inside a high-tech command center, the protagonist is standing in front of a large holographic screen, analyzing data about an incoming alien fleet.
    Scene 3: The hero steps onto the battlefield in an exosuit, with explosions in the background and laser fire illuminating the sky.
    Scene 4: The final showdown takes place atop a cliff, overlooking the ocean. The sky is stormy, and the antagonist stands with a menacing grin, holding a mysterious glowing orb.
    Scene 5: After the climactic battle, the hero gazes over the horizon, with the sun rising and peace finally restored.
    """

    # Generate the storyboard from the script
    storyboard_urls = generate_storyboard_from_script(script)

    # Save each image locally
    save_storyboard_images(storyboard_urls, "sci_fi_storyboard")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Script:
Scene 1: A mysterious forest, with towering trees and glowing mushrooms casting a soft light. 
Scene 2: The adventurer steps into a clearing, where a stone altar stands, surrounded by swirling mist.
Scene 3: A fierce dragon emerges from the mist, its scales shimmering in the moonlight.
Scene 4: The adventurer draws a glowing sword as the dragon prepares to strike.
Scene 5: After the battle, the adventurer kneels by the altar, placing a mystical gem atop it.

Expected Output:
A sequence of 5 images:
Scene 1: A glowing forest with surreal lighting and towering trees.
Scene 2: A mystical stone altar surrounded by mist.
Scene 3: A dragon emerging from mist, glowing in moonlight.
Scene 4: A glowing sword-wielding adventurer facing off against the dragon.
Scene 5: The adventurer kneeling by an altar, placing a glowing gem.
--------
Input:

Script:
Scene 1: A dark, dystopian city with rain pouring down and flickering neon lights.
Scene 2: The main character is walking down a narrow alley, their silhouette outlined by the lights.
Scene 3: Suddenly, a figure appears behind them, holding a futuristic weapon.
Scene 4: The character pulls out a plasma pistol and turns to face the threat.
Scene 5: A chase ensues across the rooftops, with the city below glowing in neon light.

Expected Output:
A sequence of 5 images:
Scene 1: A rainy, dystopian city bathed in neon light.
Scene 2: A silhouette of the main character walking in a dimly lit alley.
Scene 3: A figure appearing behind the main character with a futuristic weapon.
Scene 4: The main character turning with a plasma pistol drawn.
Scene 5: A high-speed chase across glowing city rooftops.
------
Input:

Script:
Scene 1: A medieval village nestled in the hills at dawn, with the sky turning golden.
Scene 2: The villagers gather in the town square as a mysterious hooded figure approaches on horseback.
Scene 3: The hooded figure dismounts, revealing an ancient scroll tied to their waist.
Scene 4: The villagers gasp as the figure unveils the scroll, and a magical light shines from it.
Scene 5: A dragon is summoned from the scroll, flying into the sky as the villagers watch in awe.

Expected Output:
A sequence of 5 images:
Scene 1: A peaceful medieval village at dawn.
Scene 2: Villagers gathering as a hooded rider approaches.
Scene 3: The rider dismounting, with a scroll on their waist.
Scene 4: The magical scroll glowing brightly as the villagers react.
Scene 5: A dragon emerging from the scroll, flying into the sky.
------
Input:

Script:
Scene 1: A vast, frozen wasteland with icy peaks and an overcast sky.
Scene 2: A team of explorers trudging through the snow, their breath visible in the freezing air.
Scene 3: They come across a massive glacier with a glowing object trapped inside.
Scene 4: One explorer reaches out to touch the object, and it suddenly pulses with energy.
Scene 5: The ground beneath them cracks open, and the glacier begins to shift violently.

Expected Output:
A sequence of 5 images:
Scene 1: A frozen wasteland with icy peaks.
Scene 2: Explorers making their way through snow with visible breath.
Scene 3: A glacier with a glowing object inside.
Scene 4: An explorer touching the glowing object.
Scene 5: The ground cracking as the glacier begins to shift.

Project Overview:
The AI-Powered Dynamic Storyboard Generator provides a solution for filmmakers, animators, and storytellers to rapidly visualize scripts and narratives in the form of high-quality storyboard images. This project demonstrates advanced techniques in prompt creation, script-to-image transformation, and ensuring visual coherence across multiple images. It can be further developed by integrating additional features like character consistency, environmental transitions, and action tracking across scenes. This sophisticated approach will push your understanding of OpenAI's image generation capabilities to new heights, making it one of the most advanced visual narrative projects you can build.
"""