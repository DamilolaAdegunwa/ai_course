"""
Project Title: AI-Generated Dreamscapes with Surrealism Elements
File Name: ai_generated_dreamscapes_with_surrealism.py
Description:
In this project, you will create surrealistic dreamscapes that blend abstract elements into lifelike scenarios using AI-generated imagery. The prompts will focus on merging everyday objects or environments with fantastical elements to create dreamlike images. You will explore how to use metaphors and abstract concepts in image generation, pushing your skills beyond generating simple landscapes or single-themed visuals.

By focusing on surrealism, this project will encourage you to think creatively and combine unusual elements into unique visual representations. You’ll improve your ability to work with complex, multi-faceted prompts and refine your approach to abstract concepts in AI-generated art.

Python Code:
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


# Function to generate images based on surrealism dreamscape prompts
def generate_dreamscape_image(prompt):
    """
    Generate an image based on a surrealism-themed dreamscape prompt.

    :param prompt: The prompt describing the surrealistic dreamscape.
    :return: URL of the generated image.
    """
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"  # High resolution to capture intricate surrealistic details
    )

    return response.data[0].url  # Returns the URL of the generated image


# Function to download the image from a URL and return it as a PIL Image object
def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: The URL of the image.
    :return: PIL Image object
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


# Function to save the image to a file
def save_image(image, filename):
    """
    Save the downloaded image to a file.

    :param image: PIL Image object.
    :param filename: The path and name of the file to save the image to.
    """
    image.save(filename)
    print(f"Image saved as {filename}")


# Example use cases
if __name__ == "__main__":
    # Define multiple prompts for surrealistic dreamscapes
    surreal_prompts = [
        "A floating island in the sky with giant clock towers, melting into a waterfall of time, surrounded by floating fish",
        "A forest where the trees are made of glass and the sky is filled with enormous red moons and neon stars",
        "A city where buildings are made of books, with rivers of ink flowing between the streets and giant quills as streetlamps",
        "A desert where the sand dunes are shaped like sleeping faces, and giant hourglasses replace the stars in the sky",
        "A peaceful beach where the waves are made of liquid gold, and instead of seashells, there are glowing lightbulbs scattered across the shore"
    ]

    # Generate, download, and save each surreal dreamscape image
    for i, prompt in enumerate(surreal_prompts):
        print(f"Generating image for prompt: '{prompt}'...")
        image_url = generate_dreamscape_image(prompt)
        image = download_image(image_url)
        save_image(image, f"surreal_dreamscape_{i + 1}.jpg")
"""
Multiple Example Inputs and Expected Outputs:
Input:

Prompt: "A floating island in the sky with giant clock towers, melting into a waterfall of time, surrounded by floating fish"
Expected Output:
A surreal floating island with enormous clock towers melting like Salvador Dali’s iconic imagery. The clocks form waterfalls, cascading down into a sky filled with ethereal, floating fish.
Input:

Prompt: "A forest where the trees are made of glass and the sky is filled with enormous red moons and neon stars"
Expected Output:
A fantastical glass forest, with trees reflecting their environment like mirrors. The sky is a dark canvas, with glowing red moons and vibrant neon stars creating a sci-fi fantasy landscape.
Input:

Prompt: "A city where buildings are made of books, with rivers of ink flowing between the streets and giant quills as streetlamps"
Expected Output:
A cityscape where every building is constructed from stacks of books. Ink rivers wind through the streets, and oversized quills, illuminated like streetlamps, give off a soft glow.
Input:

Prompt: "A desert where the sand dunes are shaped like sleeping faces, and giant hourglasses replace the stars in the sky"
Expected Output:
A surreal desert, with each sand dune resembling the calm face of a sleeping giant. The sky above is dotted with giant hourglasses, slowly trickling sand as if to measure the passage of time in this otherworldly landscape.
Input:

Prompt: "A peaceful beach where the waves are made of liquid gold, and instead of seashells, there are glowing lightbulbs scattered across the shore"
Expected Output:
A calm beach scene where the water glows golden in the sunlight. Instead of seashells, bright lightbulbs litter the shore, casting a surreal glow across the beach.
Project Overview:
This project will challenge your ability to create imaginative, abstract visuals by blending realistic settings with surreal and dreamlike elements. Through exploring surrealism and dreamscapes, you’ll practice constructing more complex, multi-dimensional prompts, expanding your creative range in AI image generation.
"""