# 11 https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833

"""
Next OpenAI Image Project: "AI-Driven Style Transfer and Object Inpainting with Dynamic Masking"
Project Description:
In this more advanced project, we will combine AI-driven inpainting with style transfer to create more complex image manipulations. First, we will use AI to detect and create a mask around a user-specified object in the image. We will then inpaint that masked area and replace the object with AI-generated content. Finally, we will apply a neural style transfer to the entire image to give it a new, cohesive artistic style.

Key Features:
Object Detection and Dynamic Mask Creation: Automatically detect and create a mask around a user-specified object in the image.
Inpainting with AI Replacement: Remove the detected object and replace it with AI-generated content via inpainting.
Neural Style Transfer: Apply AI-driven style transfer to the entire image to make the modified object and the rest of the image look more cohesive.
User-Defined Object and Style: Allow the user to specify the object to remove and the desired artistic style to apply.
"""

import os
import requests
import certifi
from PIL import Image, ImageDraw, ImageEnhance
from io import BytesIO
from apikey import apikey  # Assuming apikey.py contains your API key
from openai import OpenAI
import uuid
import random

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Define some artistic styles to choose from for style transfer
art_styles = ["Cubist", "Surrealist", "Watercolor", "Modernism", "Gothic"]


# Function to generate a prompt for object replacement
def generate_object_prompt(object_to_replace):
    style = random.choice(art_styles)
    prompt = f"A highly detailed {object_to_replace} in the {style} style."
    return prompt, style


# Generate an AI image based on a prompt (with literal size parameter)
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


# Function to apply inpainting by removing an object and replacing it with an AI-generated one
def apply_inpainting(image: Image, mask_area: tuple[int, int, int, int], object_to_replace):
    # Define the prompt for the object replacement
    prompt, style = generate_object_prompt(object_to_replace)
    print(f"Generating a new object with prompt: '{prompt}'")

    # Generate a new AI object to replace the region
    new_object = generate_image_from_prompt(prompt)

    # ---
    # Save the original image temporarily
    image_path = "temp_original.png"
    image.save(image_path)

    # Create a mask image
    mask = Image.new("L", image.size, 0)  # Create a black image
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_area, fill=255)  # Create a white rectangle as the mask area

    # Save mask
    mask_path = "temp_mask.png"
    mask.save(mask_path)

    # ---

    # Call OpenAI's inpainting API to replace the masked area
    inpainted_image = client.images.edit(
        image=open(image_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt=prompt,
        size="1024x1024",
        n=1,
        response_format="url"
    )
    inpainted_image_url = inpainted_image.data[0].url
    print('here is the inpainted_image_url: ' + inpainted_image_url)
    inpainted_image_response = requests.get(inpainted_image_url, verify=certifi.where())
    inpainted_image_final = Image.open(BytesIO(inpainted_image_response.content))

    # delete the temp images
    try:
        os.remove(image_path)
        os.remove(mask_path)
        print("Temporary files deleted successfully.")
    except OSError as e:
        print(f"Error deleting temporary files: {e}")

    return inpainted_image_final, style


# Function to create a mask based on the object selected by the user
def create_object_mask(image: Image, left, top, right, bottom):
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([left, top, right, bottom], fill=255)  # Masking the selected area
    return mask


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

    # Step 2: Select the area for object masking and removal
    print("You will now input the area to mask for inpainting.")
    left = int(input("Enter the left coordinate of the area to modify: "))
    top = int(input("Enter the top coordinate of the area to modify: "))
    right = int(input("Enter the right coordinate of the area to modify: "))
    bottom = int(input("Enter the bottom coordinate of the area to modify: "))
    mask_area = (left, top, right, bottom)

    # Step 3: Create a mask for the selected area
    mask = create_object_mask(image, left, top, right, bottom)

    # Step 4: Select the object to replace
    object_to_replace = input("Enter the object you want to replace (e.g., 'tree', 'building'): ")

    # Step 5: Apply inpainting to replace the object and get a new stylized image
    modified_image, style = apply_inpainting(image, mask_area, object_to_replace)

    # Step 6: Optionally adjust brightness, contrast, or sharpness
    modified_image = adjust_image(modified_image)

    # Step 7: Save the final image
    save_directory = create_save_directory()
    image_filename = f"final_image_{style}_{uuid.uuid4()}.png"
    image_path = os.path.join(save_directory, image_filename)
    modified_image.save(image_path)
    print(f"Image saved as {image_path}")
