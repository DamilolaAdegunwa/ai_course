"""
(Storybook)
Project Title: AI-Driven Storybook Illustration Generator
Description:
In this project, you will create an AI-driven generator that produces illustrations for a storybook based on user-defined story prompts. The program will allow users to input various scenes or chapters, and it will generate corresponding images that match the descriptions. This can be a fun and engaging way to visualize stories, enhancing creativity in storytelling or helping authors visualize their work.
"""
import os
import requests
from PIL import Image
from io import BytesIO
import certifi
from openai import OpenAI
from apikey import apikey  # Importing the API key from your apikey.py file

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Function to generate illustration based on a story prompt
def generate_illustration(prompt):
    """
    Generate an illustration based on the provided story prompt.
    :param prompt: The description of the scene to generate.
    :return: A PIL image of the generated illustration.
    """
    print(f"Generating illustration for: {prompt}")

    # Generate the image using the prompt
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024",  # High-quality image size
        response_format="url"
    )

    image_url = response.data[0].url
    print(f"Generated illustration: {image_url}")

    # Fetch the image from the URL
    image_response = requests.get(image_url, verify=certifi.where())
    img = Image.open(BytesIO(image_response.content))

    return img

# Function to create illustrations for a story
def create_story_illustrations(story_prompts):
    """
    Create illustrations for each scene in the story.
    :param story_prompts: A list of prompts for different scenes.
    :return: A list of generated illustration images.
    """
    illustrations = []

    for prompt in story_prompts:
        illustration = generate_illustration(prompt)
        illustrations.append(illustration)

    return illustrations

# Function to combine illustrations into a single storybook page
def combine_illustrations(illustrations, cols=2):
    """
    Combine multiple illustrations into a single storybook page.
    :param illustrations: A list of PIL images to combine.
    :param cols: Number of columns in the layout.
    :return: A single combined page image.
    """
    # Calculate the dimensions for the page
    rows = (len(illustrations) + cols - 1) // cols
    page_width = 1024 * cols
    page_height = 1024 * rows
    page_image = Image.new('RGB', (page_width, page_height), color=(255, 255, 255))

    # Paste illustrations into the page
    for index, illustration in enumerate(illustrations):
        x = (index % cols) * 1024
        y = (index // cols) * 1024
        page_image.paste(illustration.resize((1024, 1024)), (x, y))

    return page_image

# Main function to run the storybook illustration generator
def main():
    # Example story prompts
    story_prompts = [
        "A young girl discovers a magical forest filled with glowing flowers.",
        "A brave knight battles a dragon to save the kingdom.",
        "A whimsical tea party with talking animals under a rainbow.",
        "A mysterious castle surrounded by a misty lake.",
        "A spaceship exploring distant galaxies.",
        "A peaceful village during a colorful sunset."
    ]

    # Create illustrations for the story prompts
    illustrations = create_story_illustrations(story_prompts)

    # Combine illustrations into a single storybook page
    story_page = combine_illustrations(illustrations)
    story_page.show()

    # Save the storybook page image
    output_name = "storybook_page_image.png"
    story_page.save(output_name)
    print(f"Storybook page image saved as {output_name}")

if __name__ == "__main__":
    main()
"""
Key Learning Points:
Prompt Engineering: You’ll practice creating effective prompts to generate illustrations that match story themes.
Image Composition: The project includes the challenge of combining multiple images into a single page, enhancing your skills in image processing.
Creative Visualization: This project emphasizes the importance of visual storytelling and how images can enhance narrative understanding.
Example Use Cases:
Children’s Storybook: Create illustrations for scenes like "A dragon flies over a colorful village" or "A brave child finds a hidden treasure in the forest."
Fantasy Novel: Illustrate key scenes such as "A wizard casting a spell in an ancient library" or "A fierce battle between elves and goblins."
Personal Memoir: Generate images for memorable moments such as "Family gathering during a holiday" or "An unforgettable trip to the mountains."
Challenge for Further Improvement:
User Input for Story Prompts: Allow users to input their own story prompts at runtime.
Page Layout Options: Enable users to choose the number of columns or layout style for the illustrations.
Export Formats: Add functionality to export the generated storybook page in different formats (e.g., PDF, JPEG).
"""