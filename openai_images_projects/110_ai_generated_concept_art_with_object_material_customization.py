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

# Define object types, materials, and artistic styles
objects = ["Robot", "Spaceship", "Sword", "House", "Car", "Tree", "Dragon", "Castle"]
materials = ["Metal", "Wood", "Stone", "Crystal", "Glass", "Plastic", "Shiny chrome", "Rusty metal"]
styles = ["Photorealistic", "Sketch", "Cartoonish", "Futuristic", "Fantasy", "Post-apocalyptic", "Retro-futurism"]

# Streamlit UI setup
st.title("AI-Generated Concept Art with Object and Material Customization")
st.write("Create concept art by specifying objects, materials, and artistic styles for your image.")

# User input: Object, Material, Artistic Style, and Image Size
object_type = st.selectbox("Select an object to generate:", objects)
material = st.selectbox("Select a material for the object:", materials)
style = st.selectbox("Select an artistic style:", styles)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Image"):
    # Combine the inputs into a full prompt
    full_prompt = f"A {object_type.lower()} made of {material.lower()} in a {style.lower()} style"
    st.write(f"Generating image for: '{full_prompt}'")

    # Generate the image
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
    st.download_button(label="Download Image", data=img_bytes, file_name=f"{object_type.lower().replace(' ', '_')}_{material.lower().replace(' ', '_')}_{style.lower().replace(' ', '_')}.png", mime="image/png")