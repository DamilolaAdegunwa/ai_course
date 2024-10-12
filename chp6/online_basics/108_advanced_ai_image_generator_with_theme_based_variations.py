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

# Define themes and moods
themes = ["Futuristic", "Mythological", "Medieval", "Surreal", "Romantic", "Sci-Fi", "Fantasy", "Cyberpunk"]
moods = ["Euphoric", "Melancholic", "Dark", "Vibrant", "Calm", "Tranquil", "Chaotic", "Mysterious"]

# Streamlit UI setup
st.title("Advanced AI Image Generator with Theme-Based Variations")
st.write("Create AI-generated images based on themes and moods. Select a theme and mood to enrich your image generation.")

# User input: Prompt, Theme, Mood, and Image Size
prompt = st.text_input("Enter the image prompt:", "A futuristic city under the stars")
theme = st.selectbox("Select a theme:", themes)
mood = st.selectbox("Select a mood:", moods)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Image"):
    # Combine the inputs into a full prompt
    full_prompt = f"{prompt} in a {theme} theme with a {mood} mood"
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
    st.download_button(label="Download Image", data=img_bytes, file_name=f"{theme.lower().replace(' ', '_')}_{mood.lower()}.png", mime="image/png")
