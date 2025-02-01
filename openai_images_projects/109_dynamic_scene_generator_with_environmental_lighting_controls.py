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

# Define environments and lighting conditions
environments = ["Mountain", "Ocean", "Forest", "Desert", "Urban", "Island", "Snowy"]
lighting_conditions = ["Natural Daylight", "Soft Morning Light", "Dramatic Sunset", "Neon Night", "Overcast", "Harsh Noon Light", "Dappled Sunlight"]

# Streamlit UI setup
st.title("Dynamic Scene Generator with Environmental and Lighting Controls")
st.write("Create AI-generated images based on specific environments and lighting conditions for detailed scene creation.")

# User input: Prompt, Environment, Lighting, and Image Size
prompt = st.text_input("Enter the scene prompt:", "A modern skyscraper by the ocean")
environment = st.selectbox("Select an environment:", environments)
lighting = st.selectbox("Select lighting conditions:", lighting_conditions)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Image"):
    # Combine the inputs into a full prompt
    full_prompt = f"{prompt} in a {environment} environment under {lighting}"
    st.write(f"Generating image for: '{full_prompt}'")

    # Generate the image
    response = client.images.generate(prompt=full_prompt, n=1, size=size)
    image_url = response.data[0].url

    # Fetch and display the image
    img_response = requests.get(image_url,verify=certifi.where())
    img = Image.open(BytesIO(img_response.content))

    st.image(img, caption=f"Generated Image: {full_prompt}", use_column_width=True)

    # Provide an option to download the image
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    st.download_button(label="Download Image", data=img_bytes, file_name=f"{environment.lower().replace(' ', '_')}_{lighting.lower().replace(' ', '_')}.png", mime="image/png")
