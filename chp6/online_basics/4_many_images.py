# Exercise 4

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

prompt_animal = "hyper realistic image of a lion in the savannah forest"
prompt_girl = "A photorealistic portrait of a 20-year-old Japanese girl with brown curly hair and captivating blue eyes, dressed in a chic formal outfit, standing confidently in an elegant urban setting, full body shot, 8k hdr, high detailed, lot of details, high quality, she exudes a poised and sophisticated demeanor, illuminated by soft, diffused morning light, with a blurred cityscape in the background."
prompt_celeb = "Taylor Swift as a nurse"
prompt_things = "A photorealistic portrait of a gulf stream private jet!"

# Generate a random UUID for the image file
guid = uuid.uuid4()

# Animal
response_animal = client.images.generate(
    model="dall-e-3",
    prompt=prompt_animal,
    size="1024x1024",
    n=1
)

print("generating an image based on the prompt: " + prompt_animal)
animal_image_url = response_animal.data[0].url
animal_image_url_response = requests.get(animal_image_url, verify=certifi.where())
animal_image_file = Image.open(BytesIO(animal_image_url_response.content))
animal_image_file.show()
animal_image_file.save(f"images/lion_in_the_savannah_forest_{uuid}.png")

# Girl
response_girl = client.images.generate(
    model="dall-e-3",
    prompt=prompt_girl,
    size="1024x1024",
    n=1
)

print("generating an image based on the prompt: " + prompt_girl)
girl_image_url = response_girl.data[0].url
girl_image_url_response = requests.get(girl_image_url, verify=certifi.where())
girl_image_file = Image.open(BytesIO(girl_image_url_response.content))
girl_image_file.show()
girl_image_file.save(f"images/girl_{uuid}.png")

# celeb
response_celeb = client.images.generate(
    model="dall-e-3",
    prompt=prompt_celeb,
    size="1024x1024",
    n=1
)

print("generating an image based on the prompt: " + prompt_celeb)
celeb_image_url = response_celeb.data[0].url
celeb_image_url_response = requests.get(celeb_image_url, verify=certifi.where())
celeb_image_file = Image.open(BytesIO(celeb_image_url_response.content))
celeb_image_file.show()
celeb_image_file.save(f"images/celeb_{uuid}.png")

# things
response_things = client.images.generate(
    model="dall-e-3",
    prompt=prompt_things,
    size="1024x1024",
    n=1
)

print("generating an image based on the prompt: " + prompt_things)
things_image_url = response_things.data[0].url
things_image_url_response = requests.get(things_image_url, verify=certifi.where())
things_image_file = Image.open(BytesIO(things_image_url_response.content))
things_image_file.show()
things_image_file.save(f"images/things_{uuid}.png")