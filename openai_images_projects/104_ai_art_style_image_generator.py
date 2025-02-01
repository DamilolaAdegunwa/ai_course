import os
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
import requests
from PIL import Image
from io import BytesIO
import certifi
# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# List of art styles
art_styles = [
    "Impressionist", "Cubist", "Surrealist", "Realism", "Pop Art", "Abstract Expressionism",
    "Futurism", "Minimalism", "Baroque", "Romanticism", "Art Nouveau", "Dadaism", "Symbolism",
    "Neoclassicism", "Constructivism", "Renaissance", "Conceptual Art", "Post-Impressionism",
    "Expressionism", "Photorealism", "Op Art", "Bauhaus", "Street Art", "Suprematism",
    "Naïve Art", "Fauvism", "Hyperrealism", "Vorticism", "Rococo", "Lyrical Abstraction",
    "Precisionism", "De Stijl", "Tachisme", "Neo-Expressionism", "Art Deco"
]

# Create a directory called 'art_styles' to store the images
if not os.path.exists('art_styles'):
    os.makedirs('art_styles')

# Function to generate an image for each art style
def generate_image_for_style(style):
    prompt = f"An artwork in the {style} style"
    response = client.images.generate(prompt=prompt, n=1, size='1024x1024')
    image_url = response.data[0].url

    # Fetch the generated image from the URL
    img_response = requests.get(image_url, verify=certifi.where())
    img = Image.open(BytesIO(img_response.content))

    # Convert style to lowercase and replace spaces with underscores for file naming
    style_filename = style.lower().replace(" ", "_") + ".png"
    img.save(os.path.join('art_styles', style_filename))
    print(f"Saved {style_filename} in 'art_styles/' folder.")

# Generate images for all art styles
for style in art_styles:
    generate_image_for_style(style)

print("All images have been generated and saved successfully.")
