import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from offline_module import *
from PIL import Image

#### Image Generation ####
st.title("Background Remover")
model_path = ("../../Models/models--mattmdjaga--segformer_b2_clothes/"
              "snapshots/f6ac72992f938a1d0073fb5e5a06fd781f19f9a2")

model = load_model_pipeline('image-segmentation', model_path)

# File uploader with the unique key from session state
uploaded_image = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if st.button("Remove Background"):
    with st.spinner('Removing Background...'):
        st.image(uploaded_image)
        pil_image = Image.open(uploaded_image)
        result = model(images=pil_image)

        # Background
        background = result[0]['mask']
        st.image(background, caption="Background")

        # Hair
        hair = result[1]['mask']
        st.image(hair, caption="Hair")