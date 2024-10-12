import os
import certifi
import streamlit as st
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
import requests
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# List of available filters
filters = ["None", "Grayscale", "Sepia", "High Contrast", "Increase Brightness"]

# Streamlit UI setup
st.title("AI-Generated Art Gallery with Custom Filters")
st.write("Enter a prompt, generate up to 5 images, and apply custom filters.")

# User input: Prompt and number of images
prompt = st.text_input("Enter the image prompt:", "A futuristic cityscape at dusk")
num_images = st.slider("Number of images to generate:", 1, 5, 3)
size = st.selectbox("Select image size:", ["1024x1024", "512x512", "256x256"])

if st.button("Generate Images"):
    st.write(f"Generating {num_images} images for: '{prompt}'")
    images = []

    for i in range(num_images):
        response = client.images.generate(prompt=prompt, n=1, size=size)
        image_url = response.data[0].url

        # Fetch and store the images
        img_response = requests.get(image_url, verify=certifi.where())
        img = Image.open(BytesIO(img_response.content))
        images.append(img)

    # Display generated images in a gallery format
    st.write("Generated Images:")
    cols = st.columns(num_images)

    for i in range(num_images):
        with cols[i]:
            st.image(images[i], caption=f"Generated Image {i + 1}")

    # Apply filters to images
    st.write("Apply Filters to Your Images:")
    filter_choice = st.selectbox("Choose a filter to apply:", filters)

    for i, img in enumerate(images):
        st.write(f"Image {i + 1}:")
        img_with_filter = img

        if filter_choice == "Grayscale":
            img_with_filter = ImageOps.grayscale(img)
        elif filter_choice == "Sepia":
            img_with_filter = ImageOps.colorize(ImageOps.grayscale(img), "#704214", "#C0C0C0")
        elif filter_choice == "High Contrast":
            enhancer = ImageEnhance.Contrast(img)
            img_with_filter = enhancer.enhance(2.0)
        elif filter_choice == "Increase Brightness":
            enhancer = ImageEnhance.Brightness(img)
            img_with_filter = enhancer.enhance(1.5)

        st.image(img_with_filter, caption=f"Filtered Image {i + 1} ({filter_choice})", use_column_width=True)

        # Provide an option to download the filtered image
        img_bytes = BytesIO()
        img_with_filter.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        st.download_button(label=f"Download Filtered Image {i + 1}", data=img_bytes,
                           file_name=f"filtered_image_{i + 1}_{filter_choice.lower()}.png", mime="image/png")
