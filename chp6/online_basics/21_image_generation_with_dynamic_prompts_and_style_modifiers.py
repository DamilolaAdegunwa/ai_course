"""
Project Title: Image Generation with Dynamic Prompts and Style Modifiers
Description:
This exercise introduces dynamic prompt modification and style variations for each image generated from a base prompt. You will generate multiple images, each with a slight tweak in the style or theme, such as "cyberpunk," "steampunk," "minimalistic," and so on. This teaches you how to modify prompts programmatically and handle batch API calls to create a visually diverse set of images while sticking to a consistent underlying concept.

Instead of manually adjusting the prompts, the script will add style modifiers to a base prompt and generate images accordingly. You’ll also learn to handle and display these variations.
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


# Function to generate multiple images from a base prompt with style modifiers
def generate_images_with_styles(prompt, styles):
    """
    Generate multiple images from a base prompt, with each image using a different style modifier.
    :param prompt: The base text prompt.
    :param styles: A list of style modifiers (e.g., "cyberpunk", "steampunk").
    :return: A list of generated images (PIL format).
    """
    images = []

    for style in styles:
        # Combine the base prompt with the style modifier
        styled_prompt = f"{prompt}, {style} style"
        print(f"Generating image with prompt: {styled_prompt}")

        # Generate the image using the modified prompt
        response = client.images.generate(
            prompt=styled_prompt,
            n=1,
            size="1024x1024",  # Set the size to 1024x1024 for high resolution
            response_format="url"
        )

        image_url = response.data[0].url
        print(f"Generated image with style '{style}': {image_url}")

        # Fetch the image from the URL
        image_response = requests.get(image_url, verify=certifi.where())
        img = Image.open(BytesIO(image_response.content))
        images.append(img)

    return images


# Function to create a collage of styled images
def create_style_collage(images, grid_size):
    """
    Create a grid collage of images generated with different style modifiers.
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


# Main function to run the dynamic style image generation and collage creation
def main():
    # Get the base prompt from the user
    prompt = input("Enter the base prompt for generating images: ")

    # List of styles to apply to the base prompt
    styles = ["cyberpunk", "steampunk", "minimalistic", "abstract"]

    # Set grid size (for example: 2x2 grid of images with different styles)
    rows = 2
    cols = 2

    # Generate images with different styles from the base prompt
    images = generate_images_with_styles(prompt, styles)

    # Create a collage of the styled images
    collage_image = create_style_collage(images, grid_size=(rows, cols))
    collage_image.show()

    # Save the collage image
    output_name = "styled_image_collage.png"
    collage_image.save(output_name)
    print(f"Collage image saved as {output_name}")


if __name__ == "__main__":
    main()
"""
Key Learning Points:
Prompt Variations with Style: You will learn how to modify a base prompt dynamically by applying different style modifiers, such as "cyberpunk" or "minimalistic," to generate unique but related images.
Grid Layout for Stylized Images: Handle multiple variations of the base image with style and create a grid layout to visualize them in a structured collage.
Efficient API Calls for Variations: Making batch API calls with dynamic prompts and processing the results into a visual output further develops your skills in API usage and handling images in Python.
Example Use Case:
Base Prompt: "A futuristic city skyline at sunset"

Styles:

"cyberpunk"
"steampunk"
"minimalistic"
"abstract"
The script will generate four images, each depicting the base concept of a futuristic city but styled according to the chosen modifiers. The result will be a collage of all four images, showing how a single concept can be interpreted differently by adding style-based variations.

Next Challenge:
Once you're comfortable with this exercise, you can push further by:

Adding More Styles: Incorporate more complex style prompts like "baroque," "post-apocalyptic," or even more abstract styles.
Dynamic Image Sizes: Experiment with varying image sizes in a single collage.
Random Style Generation: Let the script randomly choose styles or even mix styles for more creative outputs.
This project builds on your previous experience by introducing dynamic prompt modifications and style variations, helping you explore more creative ways to generate images using the OpenAI API while also improving your prompt-engineering skills.
"""