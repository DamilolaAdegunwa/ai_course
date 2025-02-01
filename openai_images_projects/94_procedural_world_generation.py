"""
Project Title: Procedural World Generation Based on Terrain Types
File Name: procedural_world_generation.py

Project Description:
In this project, you'll generate different types of terrains (e.g., desert, forest, mountains, ocean, and tundra) using text prompts. The goal is to procedurally generate a world map by specifying the terrain types and having OpenAI generate images for each section. This exercise will allow you to explore how different environmental factors can be visualized through AI. You will generate multiple terrains, which can later be stitched into a world map.

You'll also explore prompt-tuning to capture unique characteristics of each biome, including weather patterns, lighting effects, and more advanced styles of texture generation. The project is notably more advanced as it introduces procedural control over the generation process.

Python Code:
"""
from openai import OpenAI
from apikey import apikey  # Assuming you have your apikey.py file storing the key
import os

# Set up OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate terrain images based on type
def generate_image_from_prompt(prompt):
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size='1024x1024'  # Literal size
    )
    return response.data[0].url


# Function to generate procedural terrain world
def generate_world_map(terrain_types):
    world_map = {}
    for terrain in terrain_types:
        prompt = f"An expansive {terrain} landscape with realistic textures, dynamic weather, and natural lighting"
        url = generate_image_from_prompt(prompt)
        world_map[terrain] = url
    return world_map


# Example Usage: Generating different terrains for a procedural world map
if __name__ == "__main__":
    terrains = ["desert", "forest", "mountains", "ocean", "tundra"]
    world_map = generate_world_map(terrains)

    # Output each terrain with its generated image URL
    for terrain, image_url in world_map.items():
        print(f"{terrain.capitalize()} Terrain: {image_url}")
"""
Example Inputs & Expected Outputs:
Input: terrains = ["desert", "forest", "mountains", "ocean", "tundra"] Output:

Desert Terrain: URL to desert landscape
Forest Terrain: URL to forest landscape
Mountains Terrain: URL to mountainous landscape
Ocean Terrain: URL to ocean view
Tundra Terrain: URL to snowy tundra landscape
Input: terrains = ["savannah", "volcano", "plains"] Output:

Savannah Terrain: URL to savannah with grassland and trees
Volcano Terrain: URL to an active volcano landscape
Plains Terrain: URL to wide-open grass plains
Input: terrains = ["swamp", "rainforest"] Output:

Swamp Terrain: URL to a murky swamp
Rainforest Terrain: URL to a lush rainforest with heavy vegetation
Input: terrains = ["arctic", "canyon", "wetlands"] Output:

Arctic Terrain: URL to an icy arctic environment
Canyon Terrain: URL to a red-rock canyon
Wetlands Terrain: URL to a marshy wetland
Input: terrains = ["steppe", "oasis", "coral reef"] Output:

Steppe Terrain: URL to a dry, flat steppe landscape
Oasis Terrain: URL to a desert oasis with water and palm trees
Coral Reef Terrain: URL to an underwater coral reef scene
This project allows you to explore procedural generation using text prompts, while taking control over specific types of landscapes. Each prompt will yield a unique representation of that terrain, making it suitable for building a virtual world.
"""