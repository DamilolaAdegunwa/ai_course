import os
import certifi
import streamlit as st
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
import requests
from PIL import Image, ImageDraw, ImageEnhance
from io import BytesIO
import random

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# List of habitats and environmental factors
habitats = [
    "Savannah", "Arctic tundra", "Tropical rainforest", "Desert canyon", "Mountain range",
    "Wetland", "Ocean", "Boreal forest"
]
time_of_day = ["Sunrise", "Midday", "Afternoon", "Sunset", "Night"]
weather_conditions = ["Clear", "Cloudy", "Rainy", "Snowstorm", "Foggy", "Windy"]

# Streamlit UI setup
st.title("AI Wildlife Habitat Explorer with Environmental Details")
st.write("Enter an animal prompt and select a habitat along with environmental conditions to generate a detailed wildlife scene.")

# User input: Animal, habitat, time of day, weather, and image size
animal_prompt = st.text_input("Enter the animal prompt:", "A lion resting on a rock")
habitat_choice = st.selectbox("Choose a habitat:", habitats)
time_of_day_choice = st.selectbox("Time of Day:", time_of_day)
weather_choice = st.selectbox("Weather Conditions:", weather_conditions)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Wildlife Scene"):
    st.write(f"Generating image for: '{animal_prompt}' in {habitat_choice} during {time_of_day_choice} with {weather_choice} weather")

    # Generate wildlife scene based on user input
    full_prompt = f"{animal_prompt} in {habitat_choice} during {time_of_day_choice} with {weather_choice} weather"
    response = client.images.generate(prompt=full_prompt, n=1, size=size)
    image_url = response.data[0].url

    # Fetch and display the image
    img_response = requests.get(image_url, verify=certifi.where())
    wildlife_img = Image.open(BytesIO(img_response.content))
    st.image(wildlife_img, caption=f"Generated Scene: {animal_prompt} in {habitat_choice}", use_column_width=True)

    # Adjustments based on environment
    st.write("Enhancing the scene with environmental effects")

    # Example: Adjust brightness based on time of day
    enhancer = ImageEnhance.Brightness(wildlife_img)
    if time_of_day_choice == "Sunrise" or time_of_day_choice == "Sunset":
        wildlife_img = enhancer.enhance(1.2)  # Slightly brighter for golden hour
    elif time_of_day_choice == "Night":
        wildlife_img = enhancer.enhance(0.6)  # Darker for night scenes

    # Example: Add a fog effect for certain weather conditions
    if weather_choice == "Foggy":
        overlay = Image.new('RGBA', wildlife_img.size, (255, 255, 255, 80))  # Semi-transparent white layer for fog
        wildlife_img = Image.alpha_composite(wildlife_img.convert('RGBA'), overlay)

    # Display enhanced image
    st.image(wildlife_img, caption=f"Enhanced Scene: {animal_prompt} in {habitat_choice}", use_column_width=True)

    # Provide an option to download the final image
    img_bytes = BytesIO()
    wildlife_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    st.download_button(label="Download Wildlife Scene", data=img_bytes,
                       file_name=f"wildlife_scene_{habitat_choice.lower().replace(' ', '_')}.png", mime="image/png")
