import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # C:\Users\damil\PycharmProjects\ai_course

from online_module import *
from apikey import apikey

st.title("AI Recipe Generator")

client = setup_openai(apikey)

output_format = ("""
                    <h1> Fun Title of recipe </h1>
                    <h1> Table of Contents</h1> <li> links of content </li>
                    <h1> Introduction </h1><p> dish introduction</p>
                    <h1> Country of Origin </h1><p> Country of Origin</p>
                    <h1> Ingredients </h1><li>Ingredients list </li>
                    <h1> Cooking Steps</h1><li>Cooking Steps list </li>
                    <h1> FAQ </h1><p>question answers</p>
                 """)

recipe = st.text_input("Enter your prompt", value="Pasta")
image_prompt = recipe + " realistic, cinematic"

if st.button("Create Recipe"):
    with st.spinner('Generating image...'):
        image = generate_image_openai(client, image_prompt)
        st.image(image, caption=recipe, use_column_width=True)

    with st.spinner('Generating Recipe...'):
        # Create a placeholder for the text area
        text_area_placeholder = st.markdown("", unsafe_allow_html=True)

        prompt = f" Create a detailed cooking recipe for the dish named {recipe}." \
                 f" Include preparation steps and cooking tips." \
                 f" Follow the following format {output_format}"

        generate_text_openai_streamlit(client, prompt, text_area_placeholder, html=True)

# AI – 4 Every 1  Recipe Generator