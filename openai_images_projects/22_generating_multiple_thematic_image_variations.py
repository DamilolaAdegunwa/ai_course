"""
(summary: collage app)
Project Title: Generating Multiple Thematic Image Variations
Description:
In this project, you will generate multiple images using dynamic prompts but with thematic variations. The project focuses on exploring different moods and environments by adding themes such as "in a rainy night," "in a snowy forest," or "underwater" to a base prompt. The script will generate and download images based on these themes. This will help you improve in creating diverse image prompts and handling multiple image generations programmatically.

You will also manage multiple API calls and organize the results in a visually structured way. This project builds on your previous exercise, with more attention to theme-driven variations rather than style.
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


# Function to generate multiple images with thematic prompts
def generate_images_with_themes(prompt, themes):
    """
    Generate multiple images from a base prompt with different thematic variations.
    :param prompt: The base text prompt.
    :param themes: A list of thematic modifiers (e.g., "in a rainy night", "underwater").
    :return: A list of generated images (PIL format).
    """
    images = []

    for theme in themes:
        # Combine the base prompt with the theme modifier
        themed_prompt = f"{prompt}, {theme}"
        print(f"Generating image with prompt: {themed_prompt}")

        # Generate the image using the modified prompt
        response = client.images.generate(
            prompt=themed_prompt,
            n=1,
            size="1024x1024",  # Set the size to 1024x1024 for high resolution
            response_format="url"
        )

        image_url = response.data[0].url
        print(f"Generated image with theme '{theme}': {image_url}")

        # Fetch the image from the URL
        image_response = requests.get(image_url, verify=certifi.where())
        img = Image.open(BytesIO(image_response.content))
        images.append(img)

    return images


# Function to create a thematic collage of images
def create_theme_collage(images, grid_size):
    """
    Create a grid collage of images generated with different thematic variations.
    :param images: A list of PIL images.
    :param grid_size: A tuple indicating the grid size (rows, cols).
    :return: The final collage image.
    """
    rows, cols = grid_size
    num_images = len(images)

    if num_images != rows * cols:
        raise ValueError(f"Number of images ({num_images}) doesn't match the grid size ({rows}x{cols}).")

    # Set up the size for each image in the collage
    image_width, image_height = 1024, 1024

    # Create a blank canvas for the collage
    collage_width = cols * image_width
    collage_height = rows * image_height
    collage_image = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    # Place each image in the grid
    for i, img in enumerate(images):
        # Resize the image to ensure consistency
        img = img.resize((image_width, image_height))

        # Calculate the position in the grid
        row = i // cols
        col = i % cols

        # Calculate the position where the image will be pasted
        x = col * image_width
        y = row * image_height

        # Paste the image into the collage
        collage_image.paste(img, (x, y))

    return collage_image


# Main function to run the dynamic theme image generation and collage creation
def main():
    # Get the base prompt from the user
    prompt = input("Enter the base prompt for generating images: ")

    # List of themes to apply to the base prompt
    themes = ["in a rainy night", "in a snowy forest", "underwater", "at sunset"]

    # Set grid size (for example: 2x2 grid of images with different themes)
    rows = 2
    cols = 2

    # Generate images with different thematic variations from the base prompt
    images = generate_images_with_themes(prompt, themes)

    # Create a collage of the themed images
    collage_image = create_theme_collage(images, grid_size=(rows, cols))
    collage_image.show()

    # Save the collage image
    output_name = "themed_image_collage.png"
    collage_image.save(output_name)
    print(f"Collage image saved as {output_name}")


if __name__ == "__main__":
    main()
"""
Key Learning Points:
Thematic Variations: You will modify a base prompt using thematic variations like "in a rainy night" or "underwater." This enhances your ability to generate images that portray a variety of moods and scenes based on a single core concept.
Multiple Image Generation: You will handle multiple API calls efficiently to generate images using different prompt variations.
Collage Creation: You will learn how to structure these images into a cohesive visual grid, allowing for better visualization of how themes influence the output.
Example Use Case:
Base Prompt: "A futuristic city skyline"

Themes:

"in a rainy night"
"in a snowy forest"
"underwater"
"at sunset"
The script will generate four images of the futuristic city, each depicting the city in a different environment, such as rainy or snowy. The final output will be a collage of all four images, showcasing how the city looks in different conditions.

Challenge for Further Improvement:
Advanced Theme Creation: Experiment with more specific or abstract themes like "in a parallel universe," "during a thunderstorm," or "in a dream world."
Additional Customization: Try adding color schemes or lighting conditions as modifiers to further refine the thematic prompts.
More Complex Collages: Experiment with different grid sizes or combine themes and styles in a single project to explore deeper variations.
This project focuses on generating image variations based on themes. It’s a step up from your previous exercise, offering more creative flexibility and the ability to visualize different scenarios based on the same concept.
"""