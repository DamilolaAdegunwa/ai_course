import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from offline_module import *
from PIL import Image

#### Image Generation ####
st.title("Photo Semantic Finder")
# model_path = ("../../Models/models--Salesforce--blip-image-captioning-large/" "snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90")

model_path = "Salesforce/blip-image-captioning-large"

model = load_model_pipeline('image-to-text', model_path)

# File uploader with the unique key from session state
uploaded_image = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if st.button("Generate Semantics"):
    with st.spinner('Generating Semantics...'):
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, width=300)
        with col2:
            pil_image = Image.open(uploaded_image)
            semantics = model(images=pil_image)[0]['generated_text']
            st.subheader(semantics)