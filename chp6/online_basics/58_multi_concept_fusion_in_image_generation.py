"""
Project Title: Multi-Concept Fusion in Image Generation

File Name: multi_concept_fusion_in_image_generation.py

Description: In this project, we’ll create a system that generates images by fusing multiple distinct concepts into a single prompt. The system takes multiple concept inputs (e.g., a landscape, an animal, a color scheme, etc.) and combines them intelligently to produce complex, fused visual outputs. The goal is to achieve cohesive, multi-concept images by leveraging OpenAI's image generation API.

This project allows exploration into how combining different prompts can yield visually rich images and introduces the challenge of balancing unrelated concepts.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # apikey.py contains the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate an image based on a fused prompt of multiple concepts
def generate_image_from_fused_concepts(concepts):
    # Fuse the concepts into one prompt
    fused_prompt = " and ".join(concepts)

    # Generate the image
    response = client.images.generate(
        prompt=fused_prompt,
        size="1024x1024"
    )

    # Extract and return the image URL
    return response.data[0].url


# Example inputs for generating fused concept images
if __name__ == "__main__":
    # Test case 1: Combining a natural landscape and an animal
    concepts1 = ["a mountain with waterfalls", "a tiger walking through the mist"]
    print(generate_image_from_fused_concepts(concepts1))

    # Test case 2: Combining futuristic architecture with a serene forest
    concepts2 = ["futuristic skyscrapers", "a serene forest with fog"]
    print(generate_image_from_fused_concepts(concepts2))

    # Test case 3: Blending a cosmic scene and a traditional art style
    concepts3 = ["a galaxy full of stars", "a Van Gogh painting style"]
    print(generate_image_from_fused_concepts(concepts3))

    # Test case 4: Merging technology with ancient history
    concepts4 = ["a robot with golden armor", "ancient Egyptian pyramids"]
    print(generate_image_from_fused_concepts(concepts4))

    # Test case 5: Mixing a dessert with fantasy creatures
    concepts5 = ["a vast desert", "dragons flying over dunes"]
    print(generate_image_from_fused_concepts(concepts5))
"""
Example Inputs and Expected Outputs:

Input:

Concepts: ["a mountain with waterfalls", "a tiger walking through the mist"]
Expected Output: An image of a tiger in a misty mountain landscape with waterfalls.
Input:

Concepts: ["futuristic skyscrapers", "a serene forest with fog"]
Expected Output: An image blending futuristic skyscrapers with a foggy, tranquil forest.
Input:

Concepts: ["a galaxy full of stars", "a Van Gogh painting style"]
Expected Output: A cosmic scene illustrated in the style of a Van Gogh painting.
Input:

Concepts: ["a robot with golden armor", "ancient Egyptian pyramids"]
Expected Output: A golden-armored robot standing near ancient pyramids in Egypt.
Input:

Concepts: ["a vast desert", "dragons flying over dunes"]
Expected Output: A desert scene with dragons soaring above sand dunes.
This exercise is designed to test your understanding of how to combine prompts effectively to generate visually cohesive images that blend multiple themes. You can experiment with more combinations to see how diverse the outputs can get.
"""