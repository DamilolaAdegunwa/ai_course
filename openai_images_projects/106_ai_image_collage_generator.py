import os

import certifi
import streamlit as st
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

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

# Streamlit UI setup
st.title("AI-Powered Image Collage Generator with Styles")
st.write("Enter multiple prompts and select a unique or shared art style to create a custom image collage.")

# User input: Multiple Prompts and Art Styles
num_images = st.number_input("How many images would you like to include in the collage?", min_value=2, max_value=9,
                             value=3)
prompts = [st.text_input(f"Enter prompt for image {i + 1}:") for i in range(num_images)]
style_choice = st.radio("Choose a style for all images or different styles for each?",
                        ("Same style", "Different styles"))

if style_choice == "Same style":
    style = st.selectbox("Select an art style for all images:", art_styles)
    styles = [style] * num_images
else:
    styles = [st.selectbox(f"Select art style for image {i + 1}:", art_styles) for i in range(num_images)]

size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Collage"):
    # Generate images and prepare the collage
    images = []

    for i, (prompt, style) in enumerate(zip(prompts, styles)):
        full_prompt = f"{prompt} in {style} style"
        st.write(f"Generating image {i + 1} for: '{full_prompt}'")

        response = client.images.generate(prompt=full_prompt, n=1, size=size)
        image_url = response.data[0].url

        # Fetch and save the generated image
        img_response = requests.get(image_url, verify=certifi.where())
        img = Image.open(BytesIO(img_response.content))
        images.append(img)

    # Create a collage by combining the images in a grid
    cols = int(len(images) ** 0.5) + 1  # Approximate number of columns for a grid layout
    rows = len(images) // cols + (1 if len(images) % cols != 0 else 0)

    collage_width = cols * images[0].width
    collage_height = rows * images[0].height
    collage = Image.new("RGB", (collage_width, collage_height))

    # Paste images into the collage
    for index, img in enumerate(images):
        row, col = divmod(index, cols)
        collage.paste(img, (col * img.width, row * img.height))

    # Display the final collage
    st.image(collage, caption="Generated Image Collage", use_column_width=True)

    # Provide an option to download the collage
    collage_bytes = BytesIO()
    collage.save(collage_bytes, format="PNG")
    collage_bytes.seek(0)
    st.download_button(label="Download Collage", data=collage_bytes, file_name="image_collage.png", mime="image/png")
