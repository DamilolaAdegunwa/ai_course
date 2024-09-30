"""
Project Title:
Thematic Storyboard Generation for Graphic Novels

File Name:
thematic_storyboard_generation.py

Project Description:
In this project, you'll create a script that generates a series of storyboard images based on a given theme and a set of narrative elements. The storyboard consists of a sequence of scenes, each defined by character descriptions, settings, actions, and emotions. This project allows you to explore the capabilities of the OpenAI API by dynamically generating multiple images that illustrate a coherent visual narrative. The goal is to create a mini storyboard for a graphic novel, which includes at least three panels.

Python Code:
"""
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

import os

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate a storyboard prompt based on narrative elements
def generate_storyboard_prompt(narrative_elements):
    panels = []
    for element in narrative_elements:
        prompt = f"A scene showing {element['characters']} in a {element['setting']} where they are {element['action']}. The mood is {element['emotion']}."
        panels.append(prompt)
    return panels


# Function to generate an image from the prompt
def generate_image_from_prompt(prompt):
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size='1024x1024'  # Set literal size
    )
    # Get the image URL from the response
    return response.data[0].url


# Example narrative elements for the storyboard
if __name__ == "__main__":
    narrative_elements = [
        {
            "characters": "a young knight and a wise old wizard",
            "setting": "in a mystical forest",
            "action": "discussing the quest ahead",
            "emotion": "curiosity"
        },
        {
            "characters": "the knight alone",
            "setting": "at the edge of a dark cave",
            "action": "preparing to enter",
            "emotion": "fear"
        },
        {
            "characters": "the knight battling a dragon",
            "setting": "inside the cave",
            "action": "fighting fiercely",
            "emotion": "determination"
        }
    ]

    # Generate prompts for each panel
    storyboard_prompts = generate_storyboard_prompt(narrative_elements)

    # Generate images for each panel and print the URLs
    for i, prompt in enumerate(storyboard_prompts, start=1):
        image_url = generate_image_from_prompt(prompt)
        print(f"Generated Image URL for Panel {i}: {image_url}")
"""
Example Inputs and Expected Outputs:
Input:
narrative_elements[0] = {
    "characters": "a young knight and a wise old wizard",
    "setting": "in a mystical forest",
    "action": "discussing the quest ahead",
    "emotion": "curiosity"
}
Expected Output: A URL for an image of a scene depicting a young knight and a wise old wizard in a mystical forest discussing their quest, evoking curiosity.
-------------------------------------
Input:

narrative_elements[1] = {
    "characters": "the knight alone",
    "setting": "at the edge of a dark cave",
    "action": "preparing to enter",
    "emotion": "fear"
}
Expected Output: A URL for an image showing the knight at the edge of a dark cave, preparing to enter with an expression of fear.
--------------------------------------
Input:

narrative_elements[2] = {
    "characters": "the knight battling a dragon",
    "setting": "inside the cave",
    "action": "fighting fiercely",
    "emotion": "determination"
}
Expected Output: A URL for an image of the knight fiercely battling a dragon inside a dark cave, showcasing determination.
---------------------------------------
Input:

narrative_elements[0] = {
    "characters": "two thieves",
    "setting": "in a bustling marketplace",
    "action": "planning their next heist",
    "emotion": "excitement"
}
Expected Output: A URL for an image of two thieves in a bustling marketplace plotting their next heist, filled with excitement.
----------------------------------------
Input:

narrative_elements[1] = {
    "characters": "a detective",
    "setting": "in a dimly lit office",
    "action": "examining clues",
    "emotion": "frustration"
}
Expected Output: A URL for an image of a detective in a dimly lit office examining clues, expressing frustration.
"""
