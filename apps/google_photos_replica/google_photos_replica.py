import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from offline_module import *
from PIL import Image
import uuid


# Generates a descriptive caption for the image using a pre-trained model.
def get_image_semantics(image):
    semantics = model(images=image)[0]['generated_text']
    return semantics


# Renames the image with a unique identifier (UUID) and detected semantics, then saves it.
def rename_and_save_image(image, save_path, semantics):
    new_file_name = f"{uuid.uuid4()}_{semantics}.jpg"  # Create a new file name.
    new_save_path = os.path.join(Path(save_path), new_file_name)  # Determine the full save path.
    image.save(new_save_path)  # Save the image at the new location.
    return new_save_path  # Return the new save path.


# Saves uploaded files and processes them with an image captioning model.
def save_and_process_files(uploaded_files, upload_dir):
    progress_bar = st.progress(0)  # Create a progress bar to track file processing.
    count = st.empty()  # Create a placeholder to display the count of processed files.
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)  # Open the uploaded file as an image.
        semantics = get_image_semantics(image)  # Get image semantics (descriptive caption).

        # Rename the image file based on its content and save it.
        rename_and_save_image(image, upload_dir, semantics)
        progress_bar.progress(int((i + 1) / len(uploaded_files) * 100))  # Update the progress bar.
        count.text(f"{i + 1}/{len(uploaded_files)} images processed")  # Update the count display.


# Retrieves image files from the specified directory.
def get_image_files(upload_dir):
    return [os.path.join(upload_dir, filename)  # Combine directory path with file name.
            for filename in os.listdir(upload_dir)  # List all files in the directory.
            if filename.endswith(('.png', '.jpg'))]  # Filter for image files only.


# Filters images based on a search query.
def filter_images(search_query, upload_dir):
    image_files = get_image_files(upload_dir)  # Get all image files from the directory.
    if search_query:
        keywords = search_query.lower().split()  # Split the search query into keywords.
        # Keep only files whose names contain all the keywords.
        filtered_files = [file for file in image_files if
                          all(keyword in Path(file).stem.lower() for keyword in keywords)]
        return filtered_files
    else:
        return image_files


# Displays images in a grid layout within the Streamlit app.
def display_images_in_grid(image_files):
    if image_files:
        num_cols = 3  # Number of columns in the grid.
        cols = st.columns(num_cols)  # Create columns.
        for index, file_path in enumerate(image_files):
            image = Image.open(file_path)  # Open the image file.
            cols[index % num_cols].image(image, use_column_width=True)  # Display the image in a column.


#### Image Generation ####
st.title("Google Photos Replica")
model_path = ("../../Models/models--Salesforce--blip-image-captioning-large/"
              "snapshots/2227ac38c9f16105cb0412e7cab4759978a8fd90")

model = load_model_pipeline('image-to-text', model_path)

# Ensure the directory for uploaded images exists.
upload_dir = 'uploaded_images'
os.makedirs(upload_dir, exist_ok=True)

# Initialize a unique key for the file uploader widget if it doesn't already exist.
if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = uuid.uuid4().hex

# File uploader with the unique key from session state
uploaded_images = st.file_uploader("Choose a photo", accept_multiple_files=True,
                                   type=["jpg", "jpeg", "png"], key=st.session_state['file_uploader_key'])

if uploaded_images:
    save_and_process_files(uploaded_images, upload_dir)
    st.session_state['file_uploader_key'] = uuid.uuid4().hex

# Allow users to search for images based on keywords.
search_query = st.text_input("Search Images")
# Filter and display images based on the search query.
filtered_files = filter_images(search_query, upload_dir)
display_images_in_grid(filtered_files)