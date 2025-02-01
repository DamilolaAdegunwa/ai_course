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

# Define character, setting, and emotion options
characters = ["Dragon", "Knight", "Explorer", "Young Girl", "Hero", "Villain", "Wizard", "Fox", "Monster"]
settings = ["Mountain Peak", "Forest", "Castle", "Garden", "Ocean", "Battlefield", "Village", "Desert", "Space Station"]
emotions = ["Happy", "Melancholy", "Tense", "Fearful", "Courageous", "Excited", "Curious", "Relaxed", "Wonder"]

# Streamlit UI setup
st.title("Interactive Storybook Creator with Dynamic Scene Generation")
st.write("Create storybook scenes by specifying the chapter, characters, setting, and emotional tone.")

# User input: Chapter title, characters, setting, emotion, and image size
chapter_title = st.text_input("Enter the chapter title:", "The Lonely Dragon")
character = st.selectbox("Select the main character:", characters)
setting = st.selectbox("Select the setting:", settings)
emotion = st.selectbox("Select the emotional tone of the scene:", emotions)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Scene"):
    # Combine the inputs into a full prompt
    full_prompt = f"{chapter_title}: A scene with {character.lower()} in a {setting.lower()} setting, evoking a {emotion.lower()} emotion"
    st.write(f"Generating image for: '{full_prompt}'")

    # Generate the image
    response = client.images.generate(prompt=full_prompt, n=1, size=size)
    image_url = response.data[0].url

    # Fetch and display the image
    img_response = requests.get(image_url, verify=certifi.where())
    img = Image.open(BytesIO(img_response.content))

    st.image(img, caption=f"Generated Scene: {chapter_title}", use_column_width=True)

    # Provide an option to download the image
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    st.download_button(label="Download Image", data=img_bytes, file_name=f"{chapter_title.lower().replace(' ', '_')}.png", mime="image/png")

    # Option to generate the next chapter's scene
    if st.button("Start Next Chapter"):
        st.write("Proceed to the next chapter by adding a new title, characters, setting, and emotion.")
