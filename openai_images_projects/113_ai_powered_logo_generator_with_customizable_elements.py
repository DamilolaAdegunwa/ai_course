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

# Define options for customization
industries = [
    "Technology", "Environmental Consulting", "Bakery", "Fitness",
    "Clothing", "Finance", "Healthcare", "Education", "Travel", "Food & Beverage",
    "Real Estate", "Automotive", "Entertainment", "Retail", "Non-Profit"
]

color_schemes = [
    "Red and Black", "Green and Blue", "Pastel Colors", "Black and Silver",
    "Brown and Cream", "Monochrome", "Vibrant Colors", "Earth Tones",
    "Neon Colors", "Gold and White", "Blue and Orange", "Purple and Gray"
]

icon_styles = [
    "Leaf and Wave", "Circuit Board", "Cupcake", "Dumbbell", "Retro Car",
    "Book", "Heart", "Camera", "Globe", "Gear", "Star", "Mountain", "Lightbulb"
]

logo_styles = [
    "Modern", "Minimalist", "Vintage", "Playful", "Futuristic",
    "Professional", "Bold", "Elegant", "Hand-drawn", "Geometric"
]

# Streamlit UI setup
st.title("AI-Powered Logo Generator with Customizable Elements")
st.write("Create unique and professional logos tailored to your brand's identity by customizing various elements.")

# User input: Company Name, Industry, Color Scheme, Icon Style, Tagline, Logo Style, and Image Size
company_name = st.text_input("Enter your company name:", "TechNova")
industry = st.selectbox("Select your industry:", industries)
color_scheme = st.selectbox("Select a color scheme:", color_schemes)
icon_style = st.selectbox("Select an icon style:", icon_styles)
tagline = st.text_input("Enter your tagline (optional):", "Innovate the Future")
logo_style = st.selectbox("Select a logo style:", logo_styles)
size = "1024x1024"  # Using size literally as per adjustments

if st.button("Generate Logo"):
    # Combine the inputs into a full prompt
    tagline_part = f" with the tagline '{tagline}'" if tagline else ""
    full_prompt = (
        f"A {logo_style.lower()} logo for a {industry.lower()} company named '{company_name}', "
        f"featuring a {icon_style.lower()} icon, using a {color_scheme.lower()} color scheme{tagline_part}."
    )
    st.write(f"Generating logo for: '{full_prompt}'")

    try:
        # Generate the image
        response = client.images.generate(prompt=full_prompt, n=1, size=size)
        image_url = response.data[0].url

        # Fetch and display the image
        img_response = requests.get(image_url, verify=certifi.where())
        img = Image.open(BytesIO(img_response.content))

        st.image(img, caption=f"Generated Logo: {company_name}", use_column_width=True)

        # Provide an option to download the image
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        download_filename = f"{company_name.lower().replace(' ', '_')}_logo.png"
        st.download_button(
            label="Download Logo",
            data=img_bytes,
            file_name=download_filename,
            mime="image/png"
        )
    except Exception as e:
        st.error(f"An error occurred while generating the logo: {e}")
