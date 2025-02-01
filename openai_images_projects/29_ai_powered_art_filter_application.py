"""
Project Title: AI-Powered Art Filter Application
Description:
In this project, you will develop an application that allows users to upload their images and apply various artistic filters to transform their photos into unique artworks. The application will use OpenAI's image generation capabilities to create artistic styles based on user-defined parameters, effectively functioning as an art filter generator. This project focuses on user interaction, creativity, and leveraging AI for artistic expression.
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

# Function to apply an artistic filter to an uploaded image
def apply_artistic_filter(image_path, filter_prompt):
    """
    Apply an artistic filter to an uploaded image based on a user-defined prompt.
    :param image_path: Path to the uploaded image.
    :param filter_prompt: Prompt describing the artistic style.
    :return: A PIL image of the filtered artwork.
    """
    print(f"Applying artistic filter: {filter_prompt}")

    # Read the uploaded image
    original_image = Image.open(image_path)

    # Generate the artwork using the filter prompt
    response = client.images.generate(
        prompt=filter_prompt,
        n=1,
        size="1024x1024",  # High-quality image size
        response_format="url"
    )

    image_url = response.data[0].url
    print(f"Generated filter artwork: {image_url}")

    # Fetch the filter image from the URL
    filter_response = requests.get(image_url, verify=certifi.where())
    filtered_artwork = Image.open(BytesIO(filter_response.content))

    # Combine the original image and the filtered artwork
    final_artwork = Image.blend(original_image.resize((1024, 1024)), filtered_artwork, alpha=0.5)

    return final_artwork

# Function to process multiple images with different filters
def process_images_with_filters(image_paths, filters):
    """
    Process multiple images with user-defined artistic filters.
    :param image_paths: List of paths to uploaded images.
    :param filters: List of prompts describing the artistic styles.
    :return: List of filtered artwork images.
    """
    filtered_artworks = []

    for index, image_path in enumerate(image_paths):
        filter_prompt = filters[index % len(filters)]  # Cycle through filters
        filtered_artwork = apply_artistic_filter(image_path, filter_prompt)
        filtered_artworks.append(filtered_artwork)

    return filtered_artworks

# Function to display the filtered artworks in a grid layout
def display_filtered_artworks(artworks, cols=3):
    """
    Display filtered artworks in a grid layout.
    :param artworks: A list of PIL images to display.
    :param cols: Number of columns in the layout.
    :return: A single combined image displaying all artworks.
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

# Main function to run the art filter application
def main():
    # Example paths to images to be processed (you should provide valid image paths)
    image_paths = [
        "path/to/your/image1.jpg",  # Replace with your actual image paths
        "path/to/your/image2.jpg",
        "path/to/your/image3.jpg"
    ]

    # Example artistic filters
    filters = [
        "Impressionist style painting",
        "Futuristic digital art",
        "Cubism-inspired artwork",
        "Surrealist dreamscape",
        "Abstract geometric patterns"
    ]

    # Process the images with the defined filters
    filtered_artworks = process_images_with_filters(image_paths, filters)

    # Display the filtered artworks in a gallery layout
    gallery_page = display_filtered_artworks(filtered_artworks)
    gallery_page.show()

    # Save the gallery image
    output_name = "filtered_art_gallery_image.png"
    gallery_page.save(output_name)
    print(f"Filtered art gallery image saved as {output_name}")

if __name__ == "__main__":
    main()
"""
Key Learning Points:
User Input Handling: This project emphasizes working with user-uploaded images, providing an opportunity to learn about file handling and image manipulation in Python.
Image Processing Techniques: By blending the original image with generated artwork, you will deepen your understanding of image processing techniques and effects.
Dynamic Filter Application: Explore how different artistic styles affect a single image, enhancing your skills in prompt engineering.
Example Use Cases:
Personalized Artwork: Users can take their favorite photos and transform them into unique artistic pieces for home decor or gifts.
Social Media Content: Users can generate eye-catching visuals for posts, stories, or profiles, giving their social media a creative boost.
Creative Workshops: Artists can utilize this tool to explore different artistic styles, aiding in brainstorming sessions or workshops.
Challenge for Further Improvement:
Web Interface: Develop a web-based interface for users to upload images and select filters, enhancing accessibility.
Additional Filters: Introduce more advanced filters, such as style transfer techniques or custom filter creation based on user sketches.
Batch Processing: Allow users to process multiple images at once, providing a more efficient workflow for generating filtered artworks.
This project aims to enhance your understanding of AI applications in art while providing practical skills in image manipulation and user interaction.
"""