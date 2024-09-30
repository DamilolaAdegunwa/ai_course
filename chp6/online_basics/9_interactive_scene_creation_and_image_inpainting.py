# 9 https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833 - Interactive Scene Creation and Image Inpainting

# import openai
import requests
import certifi
from io import BytesIO
from PIL import Image, ImageDraw
import os
import uuid
from openai import OpenAI
from apikey import apikey  # assuming you have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Set OpenAI API key
# os.environ['OPENAI_API_KEY'] = apikey
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate an initial image scene based on dynamic user input
def generate_scene(objects, scene_description):
    prompt = f"A {scene_description} with the following objects: {', '.join(objects)}."
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))

# Create directory for saving images
def create_save_directory():
    directory = f"generated_images/{uuid.uuid4()}"
    os.makedirs(directory, exist_ok=True)
    return directory

# Inpainting with OpenAI: Mask parts of the image and have OpenAI "paint" over it
def inpaint_image(original_image, mask_area, prompt_for_inpainting):
    # Save the original image temporarily
    original_image_path = "temp_original.png"
    original_image.save(original_image_path)

    # Create a mask image
    mask = Image.new("L", original_image.size, 0)  # Create a black image
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_area, fill=255)  # Create a white rectangle as the mask area

    # Save mask
    mask_path = "temp_mask.png"
    mask.save(mask_path)

    # Perform inpainting using OpenAI API
    #response = openai.Image.create_edit(
    response = client.images.edit(
        image=open(original_image_path, "rb"),
        mask=open(mask_path, "rb"),
        prompt=prompt_for_inpainting,
        n=1,
        size="1024x1024"
    )
    new_image_url = response.data[0].url
    new_image_response = requests.get(new_image_url, verify=certifi.where())
    new_image_file = Image.open(BytesIO(new_image_response.content))
    print('now showing the new image')
    new_image_file.show()
    return new_image_file

# Function to draw and visualize the masked area on the image (for user feedback)
def visualize_masked_area(image, mask_area):
    draw = ImageDraw.Draw(image)
    draw.rectangle(mask_area, outline="red", width=5)  # Visual feedback for mask area
    return image

# Main project flow
if __name__ == "__main__":
    # User input to generate the initial scene
    scene_description = input("Describe the scene you want (e.g., 'sunset beach', 'mountain landscape'): ")
    objects = input("Enter objects to include (comma-separated, e.g., 'a boat, a tree, a dog'): ").split(',')

    # Generate the scene with objects
    save_directory = create_save_directory()
    print(f"Generating scene: {scene_description} with objects: {', '.join(objects)}")
    base_image = generate_scene(objects, scene_description)

    # Save and display the generated image
    base_image_path = os.path.join(save_directory, "base_scene.png")
    base_image.save(base_image_path)
    base_image.show()
    print(f"Base scene saved as {base_image_path}")

    # Ask user if they want to apply inpainting to the scene
    while input("Do you want to modify (inpaint) a part of the image? (yes/no): ").lower() == "yes":
        # Get mask coordinates for the area the user wants to modify
        left = int(input("Enter the left coordinate of the area to modify: "))
        top = int(input("Enter the top coordinate of the area to modify: "))
        right = int(input("Enter the right coordinate of the area to modify: "))
        bottom = int(input("Enter the bottom coordinate of the area to modify: "))
        mask_area = (left, top, right, bottom)

        # Visualize the mask area for user confirmation
        visual_image = visualize_masked_area(base_image.copy(), mask_area)
        visual_image.show()

        # Get new prompt for the inpainted section
        new_prompt = input(f"Enter a description of what should replace the area ({mask_area}): ")

        # Apply inpainting to the image
        modified_image = inpaint_image(base_image, mask_area, new_prompt)

        # Save and display the modified image
        modified_image_path = os.path.join(save_directory, f"modified_image_{uuid.uuid4()}.png")
        modified_image.save(modified_image_path)
        modified_image.show()
        print(f"Modified image saved as {modified_image_path}")

        # Update the base image with modifications for subsequent edits
        base_image = modified_image
