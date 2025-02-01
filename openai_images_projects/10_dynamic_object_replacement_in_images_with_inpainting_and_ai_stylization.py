# 10 https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833 - "Dynamic Object Replacement in Images with Inpainting and AI Stylization"

#import openai
import requests
import certifi
from io import BytesIO
from PIL import Image, ImageEnhance, ImageDraw
import os
import uuid
import random
from openai import OpenAI
from apikey import apikey  # assuming you have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Artistic styles for prompt variation
art_styles = ["Impressionist", "Cubist", "Surrealist", "Realism", "Pop Art"]


# Function to generate a dynamic prompt for object replacement
def generate_object_prompt(object_to_replace):
    style = random.choice(art_styles)
    prompt = f"A highly detailed {object_to_replace} in the {style} style."
    return prompt, style


# Function to generate AI-generated object/image from prompt
def generate_image_from_prompt(prompt):
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="512x512",
        response_format="url"
    )
    image_url = response.data[0].url
    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))


# Function to apply inpainting by removing an area and replacing it with AI content
def apply_inpainting(image: Image, left, top, right, bottom, object_to_replace):
    # Define the inpainting region
    draw = ImageDraw.Draw(image)
    draw.rectangle([left, top, right, bottom], fill="white")

    # Generate a new AI object to replace the region
    prompt, style = generate_object_prompt(object_to_replace)
    print(f"Generating a new object with prompt: '{prompt}'")
    #new_object = generate_image_from_prompt(prompt, size=f"{right - left}x{bottom - top}")
    new_object = generate_image_from_prompt(prompt)

    # Paste the new object onto the image
    image.paste(new_object, (left, top))

    return image, style


# User-controlled image adjustment (brightness, contrast, sharpness)
def adjust_image(image):
    print("Would you like to adjust the image's brightness, contrast, or sharpness?")
    options = input("Enter 'brightness', 'contrast', 'sharpness', or 'none': ").lower()

    if options == 'brightness':
        factor = float(input("Enter a brightness factor (e.g., 1.0 for no change, >1 for brighter): "))
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
    elif options == 'contrast':
        factor = float(input("Enter a contrast factor (e.g., 1.0 for no change, >1 for higher contrast): "))
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
    elif options == 'sharpness':
        factor = float(input("Enter a sharpness factor (e.g., 1.0 for no change, >1 for sharper): "))
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(factor)

    return image


# Function to save image in a generated directory
def create_save_directory():
    directory = f"images/{uuid.uuid4()}"
    os.makedirs(directory, exist_ok=True)
    return directory


# Main project flow
if __name__ == "__main__":
    # Step 1: Load the initial image (assume user provides a path)
    image_path = input("Enter the path to your 1024x1024 image: ")
    image = Image.open(image_path)

    # Step 2: Choose coordinates for inpainting
    print("You will now input the area to modify.")
    left = int(input("Enter the left coordinate of the area to modify: "))
    top = int(input("Enter the top coordinate of the area to modify: "))
    right = int(input("Enter the right coordinate of the area to modify: "))
    bottom = int(input("Enter the bottom coordinate of the area to modify: "))

    # Step 3: Select object to replace
    object_to_replace = input("Enter the object you want to replace (e.g., 'tree', 'building'): ")

    # Step 4: Apply inpainting and replace object with AI-generated content
    modified_image, style = apply_inpainting(image, left, top, right, bottom, object_to_replace)

    # Step 5: Adjust brightness, contrast, or sharpness if needed
    modified_image = adjust_image(modified_image)

    # Step 6: Save the final image
    save_directory = create_save_directory()
    image_filename = f"final_image_{style}_{uuid.uuid4()}.png"
    image_path = os.path.join(save_directory, image_filename)
    modified_image.save(image_path)
    print(f"Image saved as {image_path}")
