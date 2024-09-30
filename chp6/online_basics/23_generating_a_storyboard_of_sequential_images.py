"""
(picture book)
Project Title: Generating a Storyboard of Sequential Images
Description:
In this project, you will create a sequence of images that tell a story based on a dynamic prompt. The goal is to generate four related images that show the progression of an event or scene. Each image represents a step in the narrative, such as "Beginning," "Middle," "Climax," and "End."

This exercise focuses on creating a visual storyboard or timeline using AI-generated images. It allows you to explore sequential storytelling with imagery, enhancing your skills in generating cohesive scenes that follow a logical or creative flow.
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


# Function to generate a sequence of images for a storyboard
def generate_storyboard(prompt, stages):
    """
    Generate a sequence of images to form a storyboard.
    :param prompt: The base text prompt.
    :param stages: A list of narrative stages (e.g., "Beginning", "Middle", "Climax", "End").
    :return: A list of generated images (PIL format).
    """
    images = []

    for stage in stages:
        # Combine the base prompt with the stage modifier
        staged_prompt = f"{prompt} ({stage})"
        print(f"Generating image with prompt: {staged_prompt}")

        # Generate the image using the modified prompt
        response = client.images.generate(
            prompt=staged_prompt,
            n=1,
            size="1024x1024",  # Use 1024x1024 for high-quality images
            response_format="url"
        )

        image_url = response.data[0].url
        print(f"Generated image for stage '{stage}': {image_url}")

        # Fetch the image from the URL
        image_response = requests.get(image_url, verify=certifi.where())
        img = Image.open(BytesIO(image_response.content))
        images.append(img)

    return images


# Function to create a horizontal storyboard of images
def create_storyboard_collage(images):
    """
    Create a horizontal collage of images for the storyboard.
    :param images: A list of PIL images.
    :return: The final collage image.
    """
    num_images = len(images)

    # Set up the size for each image in the storyboard
    image_width, image_height = 1024, 1024

    # Create a blank canvas for the storyboard
    storyboard_width = num_images * image_width
    storyboard_height = image_height
    storyboard_image = Image.new('RGB', (storyboard_width, storyboard_height), color=(255, 255, 255))

    # Place each image horizontally
    for i, img in enumerate(images):
        # Resize the image to ensure consistency
        img = img.resize((image_width, image_height))

        # Calculate the position in the horizontal storyboard
        x = i * image_width

        # Paste the image into the storyboard
        storyboard_image.paste(img, (x, 0))

    return storyboard_image


# Main function to run the storyboard image generation
def main():
    # Get the base prompt from the user
    prompt = input("Enter the base prompt for the storyboard: ")

    # List of narrative stages for the storyboard
    stages = ["Beginning", "Middle", "Climax", "End"]

    # Generate a sequence of images representing different stages of the story
    images = generate_storyboard(prompt, stages)

    # Create a horizontal collage of the storyboard images
    storyboard_image = create_storyboard_collage(images)
    storyboard_image.show()

    # Save the storyboard image
    output_name = "storyboard_image.png"
    storyboard_image.save(output_name)
    print(f"Storyboard image saved as {output_name}")


if __name__ == "__main__":
    main()
"""
Key Learning Points:
Sequential Storytelling: This project helps you create images in a sequential narrative format, each representing a different stage of a story.
Dynamic Prompt Adjustments: You'll enhance prompts with stage-specific keywords (e.g., "Beginning," "Middle") to guide the AI in generating images that follow a cohesive flow.
Horizontal Collage Creation: You'll combine the images into a horizontal storyboard that visually represents the progression of the event.
Example Use Case:
Base Prompt: "A knight's journey to rescue a princess"

Stages:

Beginning: "Setting off on a long journey"
Middle: "Encountering a dragon in the forest"
Climax: "The epic battle with the dragon"
End: "Rescuing the princess and returning home"
The script will generate four images:

Image 1: The knight setting off on the journey.
Image 2: The knight facing the dragon in the forest.
Image 3: The intense battle with the dragon.
Image 4: The knight and the princess returning to the castle.
The final result will be a horizontal storyboard showcasing the knight's journey from beginning to end.

Challenge for Further Improvement:
More Complex Storyboards: Add more stages to the story (e.g., 6 or 8 images) for a longer and more complex storyboard.
Thematic Storyboards: Try to generate sequential images with abstract or artistic themes, such as "A journey through time" or "The rise and fall of a civilization."
Vertical or Grid Collages: Instead of horizontal layouts, experiment with vertical or grid-based storyboards.
This project introduces you to sequential storytelling through AI-generated imagery, allowing you to visualize and narrate a series of events. It's more advanced than the previous exercise, focusing on cohesion between images and narrative flow.
"""