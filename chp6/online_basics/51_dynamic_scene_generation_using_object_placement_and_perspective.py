"""
Project Title: Dynamic Scene Generation Using Object Placement and Perspective
Description: This exercise focuses on dynamically generating scenes with multiple objects, arranged with perspective in mind. Users will provide prompts that describe a scene with various elements (e.g., objects, environments, and spatial relationships). The AI will generate images by interpreting the spatial organization and object sizes according to the prompt.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

def generate_image_from_prompt(prompt):
    try:
        response = client.images.generate(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example use cases:
if __name__ == "__main__":
    prompts = [
        "A park with a bench in the foreground, trees in the background, and a dog sitting next to the bench",
        "A futuristic cityscape with tall skyscrapers in the distance and a drone flying close to the camera",
        "A kitchen scene with a table full of food in the foreground and a window showing a garden in the background",
        "A beach scene with seashells in the foreground and people surfing in the far distance",
        "A medieval castle on a hill with mountains in the background and a knight standing near the castle gate"
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        image_url = generate_image_from_prompt(prompt)
        if image_url:
            print(f"Generated Image URL: {image_url}\n")
        else:
            print("Failed to generate image.\n")
"""
Example Input(s) with Expected Output(s):
Input:
Prompt: "A park with a bench in the foreground, trees in the background, and a dog sitting next to the bench"
Expected Output:
A generated image URL showing a park scene, with a clear focus on the bench and dog in the foreground, while trees appear in the distance.

Input:
Prompt: "A futuristic cityscape with tall skyscrapers in the distance and a drone flying close to the camera"
Expected Output:
A generated image URL with a futuristic skyline far away and a drone appearing much larger in the foreground due to perspective.

Input:
Prompt: "A kitchen scene with a table full of food in the foreground and a window showing a garden in the background"
Expected Output:
A generated image URL of a kitchen with detailed food items on the table and a window in the background framing a garden view.

Input:
Prompt: "A beach scene with seashells in the foreground and people surfing in the far distance"
Expected Output:
A generated image URL where the seashells are clearly visible close up, and tiny figures surfing are visible in the distance.

Input:
Prompt: "A medieval castle on a hill with mountains in the background and a knight standing near the castle gate"
Expected Output:
A generated image URL featuring a medieval castle at the center, with distant mountains and a knight placed near the castle’s gate.

File Name:
dynamic_scene_generation_using_object_placement_and_perspective.py
"""