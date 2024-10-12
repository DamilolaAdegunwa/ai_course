# Exercise 5

import requests
import certifi
from io import BytesIO
from PIL import Image
from enum import Enum
import uuid

# ---
from openai import OpenAI
from apikey import apikey
import os

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()
# ---

prompt_array = ["animal", "girl", "celeb", "things"]


class Color(Enum):
    animal = 1
    girl = 2
    celeb = 3
    things = 4


# Prompts
prompts = {
    "animal": "hyper realistic image of a lion in the savannah forest",
    "girl": "A photorealistic portrait of a 20-year-old Japanese girl with brown curly hair and captivating blue eyes, dressed in a chic formal outfit, standing confidently in an elegant urban setting, full body shot, 8k hdr, high detailed, lot of details, high quality, she exudes a poised and sophisticated demeanor, illuminated by soft, diffused morning light, with a blurred cityscape in the background.",
    "celeb": "Taylor Swift as a nurse",
    "things": "A photorealistic portrait of a Gulfstream private jet!"
}


# Function to generate and save image
def generate_image(category, prompt):
    print(f"Generating {category} image...")

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1
    )

    image_url = response.data[0].url
    image_response = requests.get(image_url, verify=certifi.where())

    # Open and save the image
    image_file = Image.open(BytesIO(image_response.content))
    image_file.show()
    # Generate a random GUID
    guid = uuid.uuid4()
    #image_file.save(f"{category}_image.png")
    # Print statement
    print(f"Generated UUID: {guid}")

    # Save the image with the GUID in the filename
    image_file.save(f"images/{category}_image_{guid}.png")
    print(f"{category.capitalize()} image saved as {category}_image.png")

# Loop through the prompt dictionary to generate all images
for category, prompt in prompts.items():
    generate_image(category, prompt)
