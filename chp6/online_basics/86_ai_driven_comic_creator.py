"""
Absolutely! Let's push the boundaries of what you've done so far with a more complex and dynamic project.

Project Title:
AI-Driven Generative Comic Creator with Dynamic Scene Transitioning

File Name:
ai_driven_comic_creator.py

Project Description:
In this advanced project, you will create a fully AI-driven comic book generator that not only generates individual scenes but also manages transitions between scenes based on character actions and emotions. The project uses the OpenAI API to dynamically generate comic panels, build scenes, and create a narrative flow with multiple levels of complexity, such as character evolution, mood shifts, and contextual continuity.

In this project, you will write a Python script that simulates dynamic storytelling in a comic format. The script accepts a list of scenes, where each scene contains details of characters, their actions, and the environment. The unique challenge in this project is generating scene transitions that evolve based on what happened in the previous scene, simulating a continuous storyline.

Additionally, the project incorporates AI-based stylistic variations, so that the same characters can be depicted differently across scenes depending on emotional shifts or character development (e.g., the character looks braver after overcoming a challenge). The key goal is to make this flow dynamic, making each scene visually connected but different in mood, color, and action.

Python Code:
"""
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key
import os

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate a storyboard prompt based on dynamic narrative elements
def generate_storyboard_prompt(scene, previous_scene=None):
    base_prompt = f"A scene showing {scene['characters']} in a {scene['setting']}. The characters are {scene['action']}. The mood is {scene['emotion']}."

    # If there's a previous scene, add continuity hints or transitions
    if previous_scene:
        base_prompt += f" Transition from the previous scene where {previous_scene['action']} took place in a {previous_scene['setting']}, and the mood was {previous_scene['emotion']}."

    return base_prompt


# Function to generate the AI comic image for a panel
def generate_image_from_prompt(prompt):
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size='1024x1024'  # Set literal size for high-quality comic panels
    )
    # Return the generated image URL
    return response.data[0].url


# Function to manage transitions and scene evolution
def generate_comic_panels(narrative):
    comic_panels = []
    previous_scene = None

    # Iterate through each scene and generate the corresponding comic panel
    for scene in narrative:
        prompt = generate_storyboard_prompt(scene, previous_scene)
        image_url = generate_image_from_prompt(prompt)
        comic_panels.append((scene, image_url))

        # Set the current scene as the previous scene for the next iteration
        previous_scene = scene

    return comic_panels


# Example narrative with transitions and emotional evolution
if __name__ == "__main__":
    narrative = [
        {
            "characters": "a fearless heroine",
            "setting": "a bustling cyberpunk city",
            "action": "racing through the streets on a futuristic bike",
            "emotion": "determination"
        },
        {
            "characters": "the heroine and her ally",
            "setting": "an underground hideout",
            "action": "discussing their next move while scanning a holographic map",
            "emotion": "strategic"
        },
        {
            "characters": "the heroine facing a robotic antagonist",
            "setting": "on top of a neon-lit skyscraper",
            "action": "fighting in an intense battle",
            "emotion": "desperation"
        },
        {
            "characters": "the heroine, victorious but scarred",
            "setting": "the destroyed rooftop with the city skyline in the background",
            "action": "reflecting on the battle",
            "emotion": "relief"
        }
    ]

    # Generate comic panels with scene transitions
    comic_panels = generate_comic_panels(narrative)

    # Print out each generated image URL for review
    for i, (scene, image_url) in enumerate(comic_panels, start=1):
        print(f"Panel {i} for scene '{scene['action']}': {image_url}")
"""
Key Aspects of the Project:
Dynamic Scene Transitions: Each scene is generated with context from the previous one, allowing for a fluid storytelling process. The transition between scenes will impact the generated image’s content, making it feel like a continuous story rather than disjointed panels.

Emotional Evolution: The script is designed to handle character emotions as they evolve, which means a character’s appearance and mood will reflect what they’ve been through, ensuring consistency in the comic's visual tone.

Stylized Comic Panels: Depending on the scene’s emotional content and narrative context, the comic panel could have different art styles, e.g., darker shades for a tense moment or bright, vivid colors for a heroic victory.

Scalability: You can scale this project up by adding more scenes or creating full story arcs with multiple transitions, making it applicable for generating a complete, multi-page comic book.

Example Inputs and Expected Outputs:
Input:

python
Copy code
narrative[0] = {
    "characters": "a fearless heroine",
    "setting": "a bustling cyberpunk city",
    "action": "racing through the streets on a futuristic bike",
    "emotion": "determination"
}
Expected Output: A URL of an image showing a futuristic city with the heroine racing on a bike, visually emphasizing determination.

Input:

python
Copy code
narrative[1] = {
    "characters": "the heroine and her ally",
    "setting": "an underground hideout",
    "action": "discussing their next move while scanning a holographic map",
    "emotion": "strategic"
}
Expected Output: A URL showing the heroine and an ally in an underground lair, scanning a holographic map with a strategic mood.

Input:

python
Copy code
narrative[2] = {
    "characters": "the heroine facing a robotic antagonist",
    "setting": "on top of a neon-lit skyscraper",
    "action": "fighting in an intense battle",
    "emotion": "desperation"
}
Expected Output: A URL of the heroine fighting a robotic antagonist on a skyscraper, with a dark, desperate mood.

Input:

python
Copy code
narrative[3] = {
    "characters": "the heroine, victorious but scarred",
    "setting": "the destroyed rooftop with the city skyline in the background",
    "action": "reflecting on the battle",
    "emotion": "relief"
}
Expected Output: A URL of the heroine standing on a destroyed rooftop, with a sense of relief after the battle.

Input:

python
Copy code
narrative[0] = {
    "characters": "a space explorer",
    "setting": "a derelict spaceship floating in space",
    "action": "examining alien artifacts",
    "emotion": "curiosity"
}
Expected Output: A URL showing a space explorer in a derelict spaceship, inspecting alien artifacts with a curious expression.

This project incorporates advanced AI features such as context-based transitions, emotional shifts, and dynamic storytelling. It challenges you to create a coherent narrative using generated visuals while ensuring continuity between scenes and the emotional development of characters.
"""