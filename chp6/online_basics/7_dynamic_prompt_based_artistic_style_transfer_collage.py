# Exercise 7 - https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833 - Project: Dynamic Prompt-Based Artistic Style Transfer Collage

#import openai
import requests
import certifi
from io import BytesIO
from PIL import Image, ImageOps
import os
import uuid
import random
from openai import OpenAI
from apikey import apikey  # assuming you have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


## Set OpenAI API key
#openai.api_key = os.getenv("OPENAI_API_KEY")



# Artistic styles to experiment with
art_styles = ["pablo picasso", "Vincent van Gogh", "Monet", "Leonardo da Vinci", "Gala Dalí", "Salvador Dalí", "René Magritte", "Joan Miró", "Frida Kahlo", "Max Ernst", "Marcel Duchamp", "Luis Buñuel", "André Breton", "Claude Monet", "Diego Velázquez", "Giorgio de Chirico", "Johannes Vermeer", "Federico García Lorca", "Henri Matisse", "Rembrandt van Rijn", "Hieronymus Bosch", "Man Ray", "Paul Cézanne", "Gustav Klimt", "Michelangelo", "Raphael", "Titian", "Botticelli", "Caravaggio", "Vermeer", "Edouard Manet", "Georges Seurat", "Edvard Munch", "Henri Matisse", "Jackson Pollock", "Andy Warhol"]

# Prompt base
base_prompt = "A portrait of a {subject}, in the style of {artist}."

# Dynamic prompt generation function
def generate_prompt(subject):
    style = random.choice(art_styles)
    return base_prompt.format(subject=subject, artist=style), style

# Image generation function using OpenAI's DALL-E
def generate_image(prompt):
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    image_url = response.data[0].url
    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))

# Create directory for saving images
def create_save_directory():
    directory = f"images/{uuid.uuid4()}"
    os.makedirs(directory, exist_ok=True)
    return directory

# Create collage from generated images
def create_collage(images, save_directory):
    # 2x2, 3x3, etc.
    # you can let the user decide the dimension
    collage_width = images[0].width * 2
    collage_height = images[0].height * 2
    collage_image = Image.new('RGB', (collage_width, collage_height))

    positions = [(0, 0), (images[0].width, 0), (0, images[0].height), (images[0].width, images[0].height)]

    for i, img in enumerate(images):
        img_resized = img.resize((images[0].width, images[0].height))
        collage_image.paste(img_resized, positions[i])

    collage_path = os.path.join(save_directory, "artistic_collage.png")
    collage_image.save(collage_path)
    print(f"Collage saved as {collage_path}")

# Main project flow
if __name__ == "__main__":
    subject = input("Enter the subject of your image (e.g., cat, landscape, person): ")

    # Create directory for saving images
    save_directory = create_save_directory()

    # Generate and apply artistic style transfer for 4 different images
    generated_images = []
    for _ in range(4):
        prompt, style = generate_prompt(subject)
        print(f"Generating image with prompt: '{prompt}'")

        # Generate the image
        image = generate_image(prompt)

        # Apply a filter or style adjustment if needed (you can apply custom filters or manipulations here)
        styled_image = ImageOps.autocontrast(image)

        # Save the individual styled image
        image_filename = f"{style}_{uuid.uuid4()}.png"
        image_path = os.path.join(save_directory, image_filename)
        styled_image.save(image_path)
        print(f"Image saved as {image_path}")

        # Append to list of generated images for collage
        generated_images.append(styled_image)

    # Create a collage from the generated images
    create_collage(generated_images, save_directory)
