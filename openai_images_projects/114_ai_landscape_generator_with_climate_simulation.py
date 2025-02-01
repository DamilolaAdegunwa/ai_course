import os
import certifi
import streamlit as st
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
import requests
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFilter
from io import BytesIO
import random

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Available climate effects
climates = ["None", "Sunny", "Cloudy", "Rainy", "Snowy"]

# Streamlit UI setup
st.title("AI Landscape Generator with Climate Simulation")
st.write("Enter a prompt to generate a landscape, and apply different climate effects.")

# User input: Prompt and climate selection
prompt = st.text_input("Enter the landscape prompt:", "A forest in the mountains")
climate_choice = st.selectbox("Choose a climate effect:", climates)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Landscape"):
    st.write(f"Generating landscape for: '{prompt}'")

    # Generate landscape image
    response = client.images.generate(prompt=prompt, n=1, size=size)
    image_url = response.data[0].url

    # Fetch and display the image
    img_response = requests.get(image_url, verify=certifi.where())
    img = Image.open(BytesIO(img_response.content))
    st.image(img, caption="Generated Landscape", use_column_width=True)

    # Apply climate effect
    st.write(f"Applying {climate_choice} effect")
    if climate_choice != "None":
        if climate_choice == "Sunny":
            img = ImageEnhance.Brightness(img).enhance(1.3)  # Increase brightness for sunny effect
        elif climate_choice == "Cloudy":
            img = ImageEnhance.Brightness(img).enhance(0.7)  # Lower brightness for cloudy effect
            img = img.filter(ImageFilter.GaussianBlur(2))  # Add slight blur for cloudy look
        elif climate_choice == "Rainy":
            draw = ImageDraw.Draw(img)
            for _ in range(100):  # Simulate raindrops
                x, y = random.randint(0, img.width), random.randint(0, img.height)
                draw.line((x, y, x + 3, y + 10), fill=(80, 80, 80), width=1)  # Draw raindrop lines
            img = ImageEnhance.Brightness(img).enhance(0.8)  # Slightly darken for rainy effect
        elif climate_choice == "Snowy":
            draw = ImageDraw.Draw(img)
            for _ in range(100):  # Simulate snowflakes
                x, y = random.randint(0, img.width), random.randint(0, img.height)
                draw.ellipse((x, y, x + 3, y + 3), fill=(255, 255, 255))  # Draw small white dots for snowflakes
            img = ImageEnhance.Brightness(img).enhance(0.9)  # Slightly brighten for snowy effect

        st.image(img, caption=f"Landscape with {climate_choice} Effect", use_column_width=True)

        # Provide an option to download the filtered image
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        st.download_button(label="Download Climate-Adjusted Image", data=img_bytes,
                           file_name=f"landscape_with_{climate_choice.lower()}.png", mime="image/png")
