"""
Project Title: Interactive Story Visualization Generator
Description:
In this project, you'll create an interactive application that generates visual representations of scenes from a user-defined story or narrative. Users can input text prompts for different scenes, and the application will generate images that depict those scenes. This project focuses on integrating storytelling with AI-generated visuals, allowing for a creative and engaging experience.
"""
import os
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from apikey import apikey  # Importing the API key from your apikey.py file

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Function to generate an image based on a scene description
def generate_scene_image(scene_description):
    """
    Generate an image for a given scene description.
    :param scene_description: A textual description of the scene.
    :return: A PIL image of the generated scene.
    """
    print(f"Generating image for scene: {scene_description}")

    # Generate the image using the scene description
    response = client.images.generate(
        prompt=scene_description,
        n=1,
        size="1024x1024",  # High-quality image size
        response_format="url"
    )

    image_url = response.data[0].url
    print(f"Generated scene artwork: {image_url}")

    # Fetch the generated image from the URL
    scene_response = requests.get(image_url)
    scene_image = Image.open(BytesIO(scene_response.content))

    return scene_image

# Function to create an interactive story visualization
def create_story_visualization(story_prompts):
    """
    Create a series of visualizations based on user-defined story prompts.
    :param story_prompts: A list of scene descriptions.
    :return: A list of PIL images for each scene.
    """
    scene_images = []

    for prompt in story_prompts:
        scene_image = generate_scene_image(prompt)
        scene_images.append(scene_image)

    return scene_images

# Function to display the generated scenes in a grid layout
def display_story_visualizations(scene_images, cols=2):
    """
    Display generated scene images in a grid layout.
    :param scene_images: A list of PIL images to display.
    :param cols: Number of columns in the layout.
    :return: A single combined image displaying all scenes.
    """
    rows = (len(scene_images) + cols - 1) // cols
    gallery_width = 1024 * cols
    gallery_height = 1024 * rows
    gallery_image = Image.new('RGB', (gallery_width, gallery_height), color=(255, 255, 255))

    # Paste scene images into the gallery
    for index, scene in enumerate(scene_images):
        x = (index % cols) * 1024
        y = (index // cols) * 1024
        gallery_image.paste(scene.resize((1024, 1024)), (x, y))

    return gallery_image

# Main function to run the story visualization generator
def main():
    # Example story prompts (you can replace these with user input)
    story_prompts = [
        "A tranquil forest with tall trees and sunlight filtering through the leaves.",
        "A futuristic city skyline at night, illuminated by neon lights.",
        "A cozy cottage in the mountains during a snowstorm.",
        "A dragon flying over a medieval castle at sunset.",
        "A serene beach with gentle waves and a colorful sunset."
    ]

    # Create visualizations for the story scenes
    scene_images = create_story_visualization(story_prompts)

    # Display the scenes in a grid layout
    gallery_page = display_story_visualizations(scene_images)
    gallery_page.show()

    # Save the gallery image
    output_name = "story_visualization_gallery.png"
    gallery_page.save(output_name)
    print(f"Story visualization gallery image saved as {output_name}")

if __name__ == "__main__":
    main()
"""
Key Learning Points:
Storytelling Integration: This project blends creative writing and visual art, showcasing how text prompts can be transformed into images.
Dynamic Scene Generation: Users learn how to generate multiple scenes based on varying descriptions, enhancing their skills in image generation.
Image Layout Design: Understand how to create a visually appealing display of multiple images.
Example Use Cases:
Children’s Storybooks: Authors can visualize scenes from their stories, aiding in the creation of illustrated storybooks.
Game Development: Game designers can generate concept art for various scenes based on narrative elements.
Social Media Engagement: Users can create engaging posts by visualizing short stories or poems, capturing audience attention with unique imagery.
Challenge for Further Improvement:
User Input Interface: Develop a simple GUI for users to enter story prompts and display images interactively.
Scene Transitions: Implement animations between scenes to create a more dynamic storytelling experience.
Narrative Complexity: Allow users to create branching narratives where different choices lead to different scenes being generated.
This project not only enhances your understanding of AI-generated imagery but also encourages creative thinking in storytelling and visual representation.
"""