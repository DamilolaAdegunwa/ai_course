import os
import certifi
import streamlit as st
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
import requests
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
from io import BytesIO
import random

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# List of dynamic background environments
backgrounds = [
    "Volcanic mountains", "Enchanted forest", "Stormy ocean",
    "Dark cave with glowing crystals", "Ruins of an ancient temple",
    "Snow-covered tundra", "Mystical desert with glowing sands", "Haunted castle"
]

# Streamlit UI setup
st.title("AI Fantasy Creature Generator with Dynamic Backgrounds")
st.write("Enter a creature prompt and select a background to generate a complete fantasy scene.")

# User input: Creature prompt and background selection
creature_prompt = st.text_input("Enter the fantasy creature prompt:", "A majestic phoenix rising from the ashes")
background_choice = st.selectbox("Choose a background environment:", backgrounds)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Fantasy Scene"):
    st.write(f"Generating fantasy creature: '{creature_prompt}' with background: '{background_choice}'")

    # Generate fantasy creature image
    response = client.images.generate(prompt=creature_prompt, n=1, size=size)
    creature_image_url = response.data[0].url

    # Fetch and display the creature image
    creature_img_response = requests.get(creature_image_url, verify=certifi.where())
    creature_img = Image.open(BytesIO(creature_img_response.content))
    st.image(creature_img, caption="Generated Creature", use_column_width=True)

    # Simulate background environment
    st.write(f"Applying {background_choice} background")

    if background_choice == "Volcanic mountains":
        background = Image.new("RGB", creature_img.size, (220, 20, 60))  # Red volcanic sky
        draw = ImageDraw.Draw(background)
        draw.ellipse([50, 50, 150, 150], fill=(255, 140, 0))  # Simulate a glowing lava pool
    elif background_choice == "Enchanted forest":
        background = Image.new("RGB", creature_img.size, (34, 139, 34))  # Forest green
        draw = ImageDraw.Draw(background)
        for _ in range(10):  # Simulate glowing mushrooms
            x, y = random.randint(0, creature_img.width), random.randint(0, creature_img.height)
            draw.ellipse((x, y, x + 30, y + 30), fill=(0, 255, 0))
    elif background_choice == "Stormy ocean":
        background = Image.new("RGB", creature_img.size, (25, 25, 112))  # Dark ocean blue
        draw = ImageDraw.Draw(background)
        for _ in range(5):  # Simulate lightning strikes
            x1, y1 = random.randint(0, creature_img.width), 0
            x2, y2 = random.randint(0, creature_img.width), creature_img.height
            draw.line((x1, y1, x2, y2), fill=(255, 255, 255), width=2)
    elif background_choice == "Dark cave with glowing crystals":
        background = Image.new("RGB", creature_img.size, (72, 61, 139))  # Dark cave purple
        draw = ImageDraw.Draw(background)
        for _ in range(15):  # Simulate glowing crystals
            x, y = random.randint(0, creature_img.width), random.randint(0, creature_img.height)
            draw.polygon([(x, y), (x+10, y+30), (x-10, y+30)], fill=(255, 0, 255))

    # Composite the creature onto the background
    final_image = Image.blend(background, creature_img, alpha=0.6)
    st.image(final_image, caption=f"Fantasy Creature with {background_choice} Background", use_column_width=True)

    # Provide an option to download the final image
    img_bytes = BytesIO()
    final_image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    st.download_button(label="Download Fantasy Scene", data=img_bytes,
                       file_name=f"fantasy_scene_{background_choice.lower().replace(' ', '_')}.png", mime="image/png")
