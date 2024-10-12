"""
Project Title: Dynamic Visual Storytelling Based on Multiple Inputs
File Name: dynamic_visual_storytelling.py

Project Description:
This project leverages OpenAI’s image generation capabilities to create dynamic visual storytelling scenes. It combines multiple input categories like characters, environments, time of day, and visual styles, and then constructs a unique image for each scene. This allows for a flexible system where complex imagery is generated based on creative combinations of elements. It challenges users to think in layers of inputs and outputs and dynamically adapts based on those inputs, resulting in a more advanced generative system.

This project can be extended for creating interactive storyboards or conceptual artwork based on the narrative provided.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # The API key is stored here.

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_story_image(character, environment, time_of_day, style):
    """
    Generates a storytelling scene image based on multiple inputs.
    :param character: The character description
    :param environment: The environment where the scene takes place
    :param time_of_day: The time of day (morning, night, sunset, etc.)
    :param style: The visual style (realistic, cartoon, abstract, etc.)
    :return: URL of the generated image
    """
    prompt = (f"A scene with {character} in a {environment} during {time_of_day}. "
              f"The visual style is {style}.")

    response = client.images.generate(
        prompt=prompt,
        n=1,
        size='1024x1024'
    )

    return response.data[0].url


# Example usage:
if __name__ == "__main__":
    # Example inputs for visual storytelling
    inputs = [
        ("a brave knight", "an enchanted forest", "sunset", "realistic"),
        ("a young wizard", "a castle on a cliff", "midnight", "cartoon"),
        ("an astronaut", "a distant planet", "dawn", "futuristic"),
        ("a pirate", "a stormy sea", "noon", "oil painting"),
        ("a detective", "a dark alley in a city", "night", "noir")
    ]

    # Generate images for each set of inputs
    for character, environment, time_of_day, style in inputs:
        image_url = generate_story_image(character, environment, time_of_day, style)
        print(f"Generated image URL: {image_url}")
"""
Example Inputs and Expected Outputs:
Input:

Character: "a brave knight"
Environment: "an enchanted forest"
Time of Day: "sunset"
Style: "realistic"
Expected Output: A realistic scene featuring a brave knight in an enchanted forest during sunset. The lighting is warm and the setting is magical with tall trees, glowing lights, and mystical animals.

Input:

Character: "a young wizard"
Environment: "a castle on a cliff"
Time of Day: "midnight"
Style: "cartoon"
Expected Output: A cartoon-style image showing a young wizard near a towering castle perched on a cliff under the stars. The atmosphere is magical, with animated effects like glowing spells.

Input:

Character: "an astronaut"
Environment: "a distant planet"
Time of Day: "dawn"
Style: "futuristic"
Expected Output: A futuristic image depicting an astronaut on an alien planet as the sun rises over unfamiliar terrain. The colors are sharp, with metallic surfaces reflecting the light of the distant star.

Input:

Character: "a pirate"
Environment: "a stormy sea"
Time of Day: "noon"
Style: "oil painting"
Expected Output: An oil-painting-style image of a pirate ship battling fierce waves in the middle of a storm. The sky is ominous, and the scene is filled with dramatic brushstrokes to emphasize the peril.

Input:

Character: "a detective"
Environment: "a dark alley in a city"
Time of Day: "night"
Style: "noir"
Expected Output: A noir-style image with a detective walking through a dark alley, with strong contrasts of light and shadow. The image feels like a scene from an old black-and-white mystery film, with rain falling and shadows creating suspense.

Summary:
This project expands on storytelling using dynamic inputs, combining characters, environments, time of day, and artistic styles. Each combination results in a unique image, offering endless possibilities for creative narratives and visual output. This exercise enhances comprehension of how multiple variables can be used to create rich and varied outputs, making the process more advanced and engaging for users looking to deepen their image-generation skills.
"""