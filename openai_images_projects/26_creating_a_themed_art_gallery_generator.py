"""
Project Title: Creating a Themed Art Gallery Generator
Description:
In this project, you'll create a program that generates a themed art gallery by producing several artworks based on a specific theme. You can choose themes like "Underwater Life," "Fantasy Landscapes," or "Futuristic Cities." The program will generate a collection of unique images, each representing different aspects of the chosen theme, and compile them into a visual gallery format. This project will help you explore how to leverage OpenAI’s image generation capabilities creatively while ensuring each artwork fits within a cohesive theme.
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


# Function to generate artwork based on a specific theme
def generate_artwork(prompt):
    """
    Generate artwork based on the provided theme prompt.
    :param prompt: The description of the theme to generate.
    :return: A PIL image of the generated artwork.
    """
    print(f"Generating artwork with prompt: {prompt}")

    # Generate the image using the prompt
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024",  # High-quality image size
        response_format="url"
    )

    image_url = response.data[0].url
    print(f"Generated artwork: {image_url}")

    # Fetch the image from the URL
    image_response = requests.get(image_url, verify=certifi.where())
    img = Image.open(BytesIO(image_response.content))

    return img


# Function to create a gallery of artworks
def create_gallery(theme, number_of_artworks):
    """
    Create a gallery of artworks based on a theme.
    :param theme: The main theme for the gallery.
    :param number_of_artworks: Number of artworks to generate for the gallery.
    :return: A list of generated artwork images.
    """
    artworks = []

    for i in range(number_of_artworks):
        prompt = f"{theme} artwork #{i + 1}"
        artwork = generate_artwork(prompt)
        artworks.append(artwork)

    return artworks


# Function to combine artworks into a single gallery image
def combine_artworks(artworks, cols=3):
    """
    Combine multiple artworks into a single image gallery.
    :param artworks: A list of PIL images to combine.
    :param cols: Number of columns in the gallery layout.
    :return: A single combined gallery image.
    """
    # Calculate the dimensions for the gallery
    rows = (len(artworks) + cols - 1) // cols
    gallery_width = 1024 * cols
    gallery_height = 1024 * rows
    gallery_image = Image.new('RGB', (gallery_width, gallery_height), color=(255, 255, 255))

    # Paste artworks into the gallery
    for index, artwork in enumerate(artworks):
        x = (index % cols) * 1024
        y = (index // cols) * 1024
        gallery_image.paste(artwork.resize((1024, 1024)), (x, y))

    return gallery_image


# Main function to run the gallery generator
def main():
    # Example theme and number of artworks to generate
    theme = "Fantasy Landscape"
    number_of_artworks = 6

    # Create a gallery of artworks
    artworks = create_gallery(theme, number_of_artworks)

    # Combine artworks into a single gallery image
    gallery_image = combine_artworks(artworks)
    gallery_image.show()

    # Save the gallery image
    output_name = "fantasy_gallery_image.png"
    gallery_image.save(output_name)
    print(f"Gallery image saved as {output_name}")


if __name__ == "__main__":
    main()
"""
Key Learning Points:
Thematic Art Generation: You'll learn to generate multiple artworks based on a common theme, improving your skills in prompt crafting.
Image Composition: The project focuses on combining various images into a single gallery format, enhancing your understanding of image manipulation.
Dynamic Input: The ability to change themes and the number of artworks allows for creative exploration and adaptability.
Example Use Cases:
Theme: "Underwater Life"

Artworks could include: "Colorful coral reef scene," "A majestic whale swimming," "Playful dolphins."
Theme: "Futuristic Cities"

Artworks could include: "A sprawling metropolis at night," "Flying cars over skyscrapers," "A futuristic park with robots."
Theme: "Enchanted Forest"

Artworks could include: "Magical creatures hidden among the trees," "A serene lake surrounded by glowing plants," "A mystical pathway leading to a fairy village."
Challenge for Further Improvement:
User Input for Themes: Allow users to input their preferred themes and number of artworks at runtime.
Gallery Customization: Implement features that let users customize the layout and size of the gallery.
Export Options: Add functionality to export the gallery in different formats (e.g., PDF, JPEG).
This project provides an opportunity to expand your creative abilities using OpenAI's image generation features while working on a more complex and engaging project.
"""