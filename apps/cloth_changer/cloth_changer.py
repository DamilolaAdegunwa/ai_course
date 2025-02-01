import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from offline_module import *
from PIL import Image

#### Image Generation ####
st.title("Cloth Changer")
model_path = ("../../Models/models--mattmdjaga--segformer_b2_clothes/"
              "snapshots/f6ac72992f938a1d0073fb5e5a06fd781f19f9a2")

model_path_inpaint = ("../../Models/models--kandinsky-community--kandinsky-2-2-decoder-inpaint/"
                      "snapshots/db790ad5cbcabed886f069ef2710774657621702")

model = load_model_pipeline('image-segmentation', model_path)
model_inpaint = load_model_inpaint(model_path_inpaint)

# File uploader with the unique key from session state
uploaded_image = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

prompt = st.text_input("Enter your prompt:", "Formal Shirt")

if st.button("Generate"):
    with st.spinner('Generating...'):
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_image)
            pil_image = Image.open(uploaded_image)
            result = model(images=pil_image)

            # Upper Clothes mask
            mask = result[3]['mask']

        with col2:
            image_changed = model_inpaint(prompt=prompt, image=pil_image, mask_image=mask,
                                          num_inference_steps=30, strength=0.2, height=768, width=768,
                                          guidance_scale=2).images[0]

            new_size = pil_image.size  # New dimensions in pixels
            resized_image = image_changed.resize(new_size)
            st.image(resized_image)