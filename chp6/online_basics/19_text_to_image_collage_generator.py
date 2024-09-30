"""
Project Title: Text-to-Image Collage Generator
Description:
In this project, we will create a Text-to-Image Collage Generator. The generator takes multiple prompts from the user, generates individual images for each prompt using the OpenAI API, and arranges them into a grid collage (e.g., 2x2, 3x3, etc.). You will learn to generate multiple images, manipulate their arrangement, and create a visually cohesive collage in a single output image.

This project advances on the previous one by:

Handling multiple images in a structured layout (collage).
Controlling the size and arrangement of the images.
Dealing with multiple API calls efficiently and learning basic image manipulation like resizing and pasting images into a collage.
"""
import os
import requests
from PIL import Image
from io import BytesIO
import certifi
from openai import OpenAI
from apikey import apikey  # Your apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate an image from a prompt
def generate_image_from_prompt(prompt):
    """
    Generate an image based on a text prompt using OpenAI's DALL·E model.
    """
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="512x512",  # Fixed size for each image in the collage
        response_format="url"
    )
    image_url = response.data[0].url
    print(f"Generated image for prompt '{prompt}': {image_url}")

    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))


# Function to create a collage of images
def create_collage(prompts, grid_size):
    """
    Create a collage of images generated from multiple prompts.
    :param prompts: A list of text prompts.
    :param grid_size: A tuple indicating the grid size (rows, cols).
    :return: The final collage image.
    """
    rows, cols = grid_size

    if len(prompts) != rows * cols:
        raise ValueError(f"Number of prompts ({len(prompts)}) doesn't match the grid size ({rows}x{cols}).")

    # Set up the size of each image in the collage
    image_width, image_height = 512, 512

    # Create a blank canvas for the collage
    collage_width = cols * image_width
    collage_height = rows * image_height
    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    # Generate and place each image in the grid
    for i, prompt in enumerate(prompts):
        img = generate_image_from_prompt(prompt)

        # Resize the image to ensure consistency in the collage
        img = img.resize((image_width, image_height))

        # Calculate position in the grid
        row = i // cols
        col = i % cols

        # Calculate the position where the image will be pasted
        x = col * image_width
        y = row * image_height

        # Paste the image into the collage
        collage_image.paste(img, (x, y))

    return collage_image


# Main function to run the collage generation
def main():
    # Get grid size from the user
    rows = int(input("Enter the number of rows for the collage: "))
    cols = int(input("Enter the number of columns for the collage: "))

    num_prompts = rows * cols
    prompts = []

    # Get prompts from the user
    for i in range(num_prompts):
        prompt = input(f"Enter prompt {i + 1}: ")
        prompts.append(prompt)

    # Generate and display the collage
    collage_image = create_collage(prompts, grid_size=(rows, cols))
    collage_image.show()

    # Save the collage image
    output_name = "collage_image.png"
    collage_image.save(output_name)
    print(f"Collage image saved as {output_name}")


if __name__ == "__main__":
    main()

"""
Key Learning Points:
Image Grid Layout: You will learn to organize images into a structured grid layout (collage), which requires positioning and pasting images in a specific arrangement.
Multiple API Requests: The project will reinforce your understanding of making multiple OpenAI API calls, handling them efficiently, and organizing the outputs into a single image.
Resizing and Pasting Images: Basic image manipulation techniques like resizing images to fit them into a grid layout and pasting them onto a blank canvas will be covered.
Example Use Case:
Create a 2x2 collage of different artistic styles by entering prompts like:

"A futuristic cityscape"
"A serene mountain lake"
"Abstract geometric shapes"
"A surreal dreamscape"
The result will be a collage of four images, each reflecting one of the prompts, arranged in a grid.

Challenges for Further Development:
Custom Collage Sizes: Allow users to define custom sizes for each image in the collage rather than fixing all images to the same size.
Advanced Layouts: Explore more complex collage arrangements like diagonal or circular layouts, where the grid is less rigid.
Interactive Collage: Allow users to drag and drop images to rearrange them before generating the final collage, offering more creative control.
This project is noticeably more advanced than the previous one because you now manage multiple images at once and focus on arranging them into a cohesive structure. The additional complexity of generating several images, handling their placement, and working with a grid introduces new challenges while still being manageable.
"""