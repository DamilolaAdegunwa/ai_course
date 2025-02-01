import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from offline_module import *

#### Image Generation ####
st.title("Coloring Page")
model_path_sdxl = (f"C:/Users/damil/PycharmProjects/ai_course/models/models--stabilityai--stable-diffusion-xl-base-1.0/"
                   "snapshots/462165984030d82259a11f4367a4eed129e94a7b")

# lora_path = (f"C:/Users/damil/PycharmProjects/ai_course/models/loras/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors")

lora_path = "artificialguybr/ColoringBookRedmond"

base, refiner = load_model_local_sdxl(model_path_sdxl, None, lora_path)

user_input = st.text_input("Enter your prompt", value="lion")
lora_trigger = ",ColoringBookAF, Coloring Book"
prompt = user_input + lora_trigger + ", low details"

# Adding a slider for the number of images
num_images = st.slider("Select number of images to generate:", 1, 10, 2)

if st.button("Generate Image"):
    with st.spinner('Generating image...'):
        progress_bar = st.progress(0)

        # Adjust for image generation
        for i in range(num_images):
            if i % 2 == 0:
                cols = st.columns(2)  # Create two columns only for even index

            # Generate image
            generated_image = generate_image_local_sdxl(base, prompt)

            # Display image in the appropriate column
            with cols[i % 2]:  # Use modulus to toggle between 0 and 1 for column index
                st.image(generated_image, channels="BGR", use_column_width=True)

            # Update progress after each image is generated and displayed
            progress = ((i + 1) / num_images)
            progress_bar.progress(int(progress * 100))