"""
Project: Themed Image Grid Generator
In this project, you will create a Python script that generates a themed grid of images based on multiple prompts. The challenge is to automate the generation of images, arrange them in a grid format, and save them as a single image file. This project is a step up in complexity, as it involves managing multiple image generation processes, combining them, and exporting the result as a final image.

Python Code
"""
import os
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_image_from_prompt(prompt):
    """
    Generate an image based on a description prompt.

    :param prompt: A string containing the image description.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="512x512"
    )
    return response.data[0].url  # Returns the URL of the generated image


def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: URL of the image.
    :return: PIL Image object.
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def create_image_grid(image_urls, grid_size=(2, 2), img_size=(512, 512)):
    """
    Arrange multiple images in a grid and save the result.

    :param image_urls: List of image URLs to include in the grid.
    :param grid_size: Tuple of (rows, columns) defining the grid layout.
    :param img_size: Tuple defining the size of each image.
    :return: PIL Image object of the grid.
    """
    rows, cols = grid_size
    width, height = img_size
    grid_img = Image.new('RGB', (cols * width, rows * height))

    for idx, url in enumerate(image_urls):
        img = download_image(url).resize(img_size)
        x = idx % cols * width
        y = idx // cols * height
        grid_img.paste(img, (x, y))

    return grid_img


def generate_themed_grid(prompts, grid_size=(2, 2)):
    """
    Generate a grid of themed images based on multiple prompts.

    :param prompts: List of prompts for generating images.
    :param grid_size: Tuple of (rows, columns) defining the grid layout.
    :return: URL of the generated grid image.
    """
    # Generate images based on prompts
    image_urls = [generate_image_from_prompt(prompt) for prompt in prompts]

    # Create the image grid
    grid_image = create_image_grid(image_urls, grid_size)

    # Save the grid image
    grid_image.save("themed_image_grid.jpg")
    print("Image grid saved as 'themed_image_grid.jpg'")


# Example use cases
if __name__ == "__main__":
    # List of prompts based on a chosen theme (e.g., fantasy theme)
    prompts = [
        "A wizard casting a spell in a dark forest",
        "A dragon flying over a mountain",
        "A warrior standing on a battlefield at sunset",
        "A mystical castle in the clouds"
    ]

    # Generate and save the themed image grid
    generate_themed_grid(prompts, grid_size=(2, 2))
"""
How to Test the Project:
Set Up Your Environment:

Make sure you have the necessary libraries installed:
bash
Copy code
pip install openai requests pillow
Configure API Key:

Ensure that you have an apikey.py file containing your OpenAI API key:
python
Copy code
apikey = 'YOUR_API_KEY'
Run the Script:

Execute the script. It will generate images based on the provided prompts, arrange them in a grid, and save the final grid as themed_image_grid.jpg.
Example Use Cases:
Storyboards: Use this to create a quick visualization of scenes in a fantasy story or game concept.
Mood Boards: You can use this to generate inspiration boards for design projects, assembling various images in a theme.
Art Exhibits: Create digital exhibits or presentations showcasing art pieces around specific themes or styles.
Educational Purposes: Visualize different concepts and themes for educational presentations, such as generating a set of historical or scientific images.
This project offers a noticeable advancement in complexity, integrating multiple AI-generated images into a final, cohesive design. It allows for various testing scenarios and provides flexible use cases that can be easily adapted to your needs.
"""