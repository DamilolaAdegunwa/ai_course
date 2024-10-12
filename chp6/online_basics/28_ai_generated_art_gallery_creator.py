"""
Project Title: AI-Generated Art Gallery Creator
Description:
In this project, you will create an AI-driven application that generates a collection of artworks based on user-defined themes or keywords. The generated images will be compiled into a virtual art gallery. This project allows for creative exploration and can serve as an inspiration tool for artists, designers, or anyone interested in visual art.

The application will accept multiple themes, generate images for each theme, and display them in a structured format, resembling an art gallery.
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

# Function to generate artwork based on a theme
def generate_artwork(theme):
    """
    Generate artwork based on the provided theme.
    :param theme: The theme or keyword for the artwork.
    :return: A PIL image of the generated artwork.
    """
    print(f"Generating artwork for theme: {theme}")

    # Generate the image using the theme
    response = client.images.generate(
        prompt=theme,
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

# Function to create an art gallery based on multiple themes
def create_art_gallery(themes):
    """
    Create an art gallery with artworks based on multiple themes.
    :param themes: A list of themes for the artworks.
    :return: A list of generated artwork images.
    """
    artworks = []

    for theme in themes:
        artwork = generate_artwork(theme)
        artworks.append(artwork)

    return artworks

# Function to display artworks in a grid layout
def display_artworks(artworks, cols=3):
    """
    Display artworks in a grid layout.
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

# Main function to run the art gallery creator
def main():
    # Example themes for the artworks
    themes = [
        "A serene landscape with mountains and a lake at sunrise.",
        "A futuristic city skyline with flying cars.",
        "An abstract representation of music and sound.",
        "A cozy library filled with books and warm light.",
        "A mythical creature in a magical forest."
    ]

    # Create artworks for the themes
    artworks = create_art_gallery(themes)

    # Display artworks in a gallery layout
    gallery_page = display_artworks(artworks)
    gallery_page.show()

    # Save the gallery image
    output_name = "art_gallery_image.png"
    gallery_page.save(output_name)
    print(f"Art gallery image saved as {output_name}")

if __name__ == "__main__":
    main()
"""
Key Learning Points:
Dynamic Theme Generation: You’ll explore how different themes can produce diverse visual outcomes, enhancing your understanding of how AI interprets various prompts.
Image Management: The project involves managing multiple images and displaying them effectively, enhancing your skills in image processing and layout design.
Artistic Exploration: This project promotes creativity and artistic expression, allowing you to visualize abstract ideas through AI-generated art.
Example Use Cases:
Creative Portfolio: Artists can generate pieces to expand their portfolios or explore new styles without manual creation.
Themed Events: Event planners can create visuals for themed parties, weddings, or exhibitions based on specific concepts (e.g., “Underwater Wonderland” or “Retro Futurism”).
Educational Tools: Teachers can use generated art to illustrate concepts in art history, culture, or even literature, providing visual aids for students.
Challenge for Further Improvement:
User Interface: Implement a simple user interface to allow users to input their themes via a graphical or web-based form.
Customization Options: Provide users with options to select different styles or color palettes for their artworks.
Save to Gallery: Add functionality to save multiple gallery pages as a PDF or another user-friendly format for sharing or printing.
This project presents a unique opportunity to enhance your skills with AI image generation, offering both creative freedom and technical challenges.
"""