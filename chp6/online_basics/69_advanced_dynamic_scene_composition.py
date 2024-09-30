"""
Project Title:
Advanced Generative Art with Dynamic Scene Composition

File Name:
advanced_dynamic_scene_composition.py

Project Description:
This project involves creating dynamically composed scenes based on a list of objects, their positions, and lighting conditions. Instead of a single prompt, this script generates more complex and structured scenes, blending different objects with customizable layouts, sky conditions, and perspectives. This exercise is designed to help you practice constructing multi-element image prompts programmatically.

The challenge involves dynamically generating detailed prompts with various parameters like object type, object position, lighting conditions, and artistic style, then generating the corresponding image using OpenAI's API.

Python Code:
"""
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

import os

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate a dynamic scene prompt based on parameters
def generate_scene(objects, lighting, sky_condition, perspective, style):
    # Construct the scene description dynamically
    scene_description = f"A {lighting} scene with a {sky_condition} sky. "
    for obj in objects:
        scene_description += f"A {obj['type']} is {obj['position']} in the frame. "

    scene_description += f"The perspective is {perspective}, and the style is {style}."
    return scene_description


# Function to generate an image from the dynamic prompt
def generate_image_from_prompt(prompt):
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size='1024x1024'  # Set literal size
    )
    # Get the image URL from the response
    return response.data[0].url


# Example test cases
if __name__ == "__main__":
    # Example 1: Dynamic scene with mountains, a tree, and a river
    objects_1 = [
        {"type": "mountain", "position": "in the background"},
        {"type": "tree", "position": "to the right"},
        {"type": "river", "position": "flowing in the foreground"}
    ]
    scene_1 = generate_scene(objects_1, lighting="sunset", sky_condition="cloudy", perspective="wide-angle",
                             style="photorealistic")
    print(f"Generated Image URL 1: {generate_image_from_prompt(scene_1)}")

    # Example 2: A futuristic city with flying cars and skyscrapers
    objects_2 = [
        {"type": "skyscraper", "position": "in the background"},
        {"type": "flying car", "position": "zooming above the streets"}
    ]
    scene_2 = generate_scene(objects_2, lighting="neon-lit night", sky_condition="clear", perspective="bird's-eye view",
                             style="cyberpunk")
    print(f"Generated Image URL 2: {generate_image_from_prompt(scene_2)}")

    # Example 3: A serene beach with a boat and seagulls
    objects_3 = [
        {"type": "boat", "position": "floating on the water"},
        {"type": "seagulls", "position": "flying in the sky"}
    ]
    scene_3 = generate_scene(objects_3, lighting="morning", sky_condition="partly cloudy", perspective="panoramic",
                             style="watercolor")
    print(f"Generated Image URL 3: {generate_image_from_prompt(scene_3)}")

    # Example 4: A medieval castle with a dragon in the sky
    objects_4 = [
        {"type": "castle", "position": "perched on a hill"},
        {"type": "dragon", "position": "flying above"}
    ]
    scene_4 = generate_scene(objects_4, lighting="moonlit", sky_condition="starry", perspective="low-angle",
                             style="fantasy art")
    print(f"Generated Image URL 4: {generate_image_from_prompt(scene_4)}")

    # Example 5: A dense forest with a river and a waterfall
    objects_5 = [
        {"type": "forest", "position": "surrounding the frame"},
        {"type": "waterfall", "position": "cascading in the center"},
        {"type": "river", "position": "flowing towards the viewer"}
    ]
    scene_5 = generate_scene(objects_5, lighting="dawn", sky_condition="misty", perspective="first-person view",
                             style="realistic")
    print(f"Generated Image URL 5: {generate_image_from_prompt(scene_5)}")
"""
Example Inputs and Expected Outputs:
Input:

objects_1 = [
    {"type": "mountain", "position": "in the background"},
    {"type": "tree", "position": "to the right"},
    {"type": "river", "position": "flowing in the foreground"}
]
generate_scene(objects_1, lighting="sunset", sky_condition="cloudy", perspective="wide-angle", style="photorealistic")

Expected Output: A URL for an image of a photorealistic wide-angle scene with a sunset over a mountain, a tree to the right, and a river flowing in the foreground under a cloudy sky.
-------------------------------------
Input:

objects_2 = [
    {"type": "skyscraper", "position": "in the background"},
    {"type": "flying car", "position": "zooming above the streets"}
]
generate_scene(objects_2, lighting="neon-lit night", sky_condition="clear", perspective="bird's-eye view", style="cyberpunk")

Expected Output: A URL for an image of a cyberpunk city with flying cars at night under a clear sky from a bird's-eye view.
-------------------------------------
Input:

objects_3 = [
    {"type": "boat", "position": "floating on the water"},
    {"type": "seagulls", "position": "flying in the sky"}
]
generate_scene(objects_3, lighting="morning", sky_condition="partly cloudy", perspective="panoramic", style="watercolor")

Expected Output: A URL for an image of a peaceful beach scene in watercolor style with a boat on the water and seagulls flying above under a partly cloudy sky.
--------------------------------------
Input:

objects_4 = [
    {"type": "castle", "position": "perched on a hill"},
    {"type": "dragon", "position": "flying above"}
]
generate_scene(objects_4, lighting="moonlit", sky_condition="starry", perspective="low-angle", style="fantasy art")

Expected Output: A URL for an image of a fantasy-style castle under a starry sky with a dragon flying overhead, lit by moonlight.
--------------------------------------
Input:

objects_5 = [
    {"type": "forest", "position": "surrounding the frame"},
    {"type": "waterfall", "position": "cascading in the center"},
    {"type": "river", "position": "flowing towards the viewer"}
]
generate_scene(objects_5, lighting="dawn", sky_condition="misty", perspective="first-person view", style="realistic")

Expected Output: A URL for an image of a realistic dawn scene in a forest, with a waterfall cascading in the center and a river flowing towards the viewer under a misty sky.
----------------------------------------

This exercise builds on your prior work by adding complexity to prompt generation, combining various elements in scenes.    
"""