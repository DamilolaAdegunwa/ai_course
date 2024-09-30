"""
Project Title: Thematic Image Collage Generator
Description:
In this project, you will build a thematic image collage generator. The user will input a theme, and the program will generate a set of images based on sub-themes related to the main theme. These images will then be combined into a single image collage that visually represents the user's chosen theme.

This project is more advanced than the previous one because:

Multiple Image Generations: It generates multiple images based on sub-themes.
Image Collage Creation: You'll combine the images programmatically to create a cohesive collage.
Handling Multiple Prompts: The exercise will require efficient handling of multiple AI-generated prompts and organizing the results into a combined output.
Key Features:
Multi-Image Generation: For each theme, several images related to sub-themes will be generated.
Image Collage Creation: The images will be combined into a grid-based collage to visually capture the essence of the theme.
Customization: The user can select the number of images they want in their collage
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
        size="512x512",  # Static size for the images in the collage
        response_format="url"
    )
    image_url = response.data[0].url
    print(f"Generated image for prompt '{prompt}': {image_url}")

    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))


# Function to create a collage from multiple images
def create_image_collage(images, grid_size=(2, 2), image_size=(512, 512)):
    """
    Create a collage from multiple images.
    :param images: A list of PIL images.
    :param grid_size: The number of images horizontally and vertically (rows, cols).
    :param image_size: Size of each image in the collage.
    :return: A PIL image object representing the collage.
    """
    collage_width = grid_size[1] * image_size[0]
    collage_height = grid_size[0] * image_size[1]

    collage_image = Image.new('RGB', (collage_width, collage_height))

    for index, image in enumerate(images):
        row = index // grid_size[1]
        col = index % grid_size[1]
        position = (col * image_size[0], row * image_size[1])
        collage_image.paste(image, position)

    return collage_image


# Function to generate images based on sub-themes and create a collage
def generate_thematic_collage(main_theme, sub_themes):
    """
    Generate images for each sub-theme and combine them into a collage.
    :param main_theme: The main theme of the collage.
    :param sub_themes: A list of sub-themes to generate images for.
    :return: A collage image.
    """
    images = []
    for sub_theme in sub_themes:
        full_prompt = f"{main_theme} - {sub_theme}"
        image = generate_image_from_prompt(full_prompt)
        images.append(image)


    grid_size = (len(images) // 2, 2)  # Assuming 2 images per row, you can adjust
    collage_image = create_image_collage(images, grid_size=grid_size)

    return collage_image


# Main function to run the collage generation
def main():
    main_theme = input("Enter the main theme for your collage: ")
    sub_theme_count = int(input("How many sub-themes do you want to generate? "))

    sub_themes = []
    for i in range(sub_theme_count):
        sub_theme = input(f"Enter sub-theme {i + 1}: ")
        sub_themes.append(sub_theme)

    print('after the for loop')
    # Generate the thematic collage
    collage_image = generate_thematic_collage(main_theme, sub_themes)
    print('after the generate_thematic_collage method')
    # Show the collage
    collage_image.show(title=f"Collage for {main_theme}")


if __name__ == "__main__":
    main()

"""
Key Learning Points:
Sub-Themes and Dynamic Image Generation: You are generating images for sub-themes that relate to a main theme. Each image is distinct but connected by the overarching idea.
Image Manipulation: You will learn how to combine images into a grid-based collage, using basic image manipulation techniques.
User Input and Control: The user specifies the theme, the number of sub-themes, and the descriptions for each sub-theme, allowing for a dynamic and customizable experience.
Explanation:
Main Theme and Sub-Themes: The user selects a main theme, and multiple sub-themes are used to generate related images. This increases the complexity as multiple prompts are dynamically created.
Collage Creation: The images generated are placed in a grid-like pattern to create a collage. This introduces basic image manipulation with the Pillow library.
Grid Size Flexibility: The grid size can be adjusted to handle more or fewer images, and the code calculates the placement of each image dynamically.
Advanced Elements:
Collage Grid: You will need to understand how to paste multiple images into a larger blank canvas, adjusting the placement based on the image index.
Dynamic Prompt Generation: The prompts are not hard-coded; they are generated dynamically based on user input, which makes the project more flexible.
Handling Multiple Images: Managing a list of images and ensuring they fit properly into a grid introduces new complexity.
Additional Challenges:
Grid Size Customization: Add user input to customize how many rows and columns they want in their collage.
Add Titles: Use ImageDraw from Pillow to add the name of each sub-theme below the corresponding image in the collage.
Thematic Filters: Apply different color filters or effects to each image in the collage to give a unique aesthetic to the entire composition.
Advanced Use Case:
Once comfortable with this, you could extend it to create multi-level collages, where each sub-theme could be further divided into more specific prompts, generating a larger and more complex collage.

This exercise pushes the complexity of handling multiple images, dynamically creating content, and organizing visual elements, building on what you've already learned while introducing new challenges in terms of image manipulation and grid-based layouts.
"""