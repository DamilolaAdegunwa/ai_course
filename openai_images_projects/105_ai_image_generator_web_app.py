import os

import certifi
import streamlit as st
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
import requests
from PIL import Image
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
st.title("AI Image Generator with Artistic Styles")
st.write("Enter a prompt and select an art style to generate an image.")

# User input: Prompt and Art Style
prompt = st.text_input("Enter the image prompt:", "A beautiful sunset over a calm sea")
style = st.selectbox("Select an art style:", art_styles)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Image"):
    # Generate image based on user input
    full_prompt = f"{prompt} in {style} style"
    st.write(f"Generating image for: '{full_prompt}'")

    response = client.images.generate(prompt=full_prompt, n=1, size=size)
    image_url = response.data[0].url

    # Fetch and display the image
    img_response = requests.get(image_url, verify=certifi.where())
    img = Image.open(BytesIO(img_response.content))

    st.image(img, caption=f"Generated Image: {full_prompt}", use_column_width=True)

    # Provide an option to download the image
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    st.download_button(label="Download Image", data=img_bytes, file_name=f"{style.lower().replace(' ', '_')}.png",
                       mime="image/png")