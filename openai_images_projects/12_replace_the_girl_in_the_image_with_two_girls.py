import os
import requests
import certifi
from io import BytesIO
from PIL import Image, ImageDraw
from openai import OpenAI
from apikey import apikey  # Load your API key from a local file

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

def generate_look_alike(prompt):
    # Call the OpenAI API to generate the look-alike
    #file = open(r"C:\Users\damil\PycharmProjects\ai_course\tut\openai_images_projects\images\2d98d1f9-cec0-4e1b-b03d-35fb0b3c75fb\generated_image_2d98d1f9-cec0-4e1b-b03d-35fb0b3c75fb.png", "rb")
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="url"
    )
    image_url = response.data[0].url
    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))

def place_two_girls(image: Image):
    # Save the original image temporarily
    image_path = "temp_image.png"
    image.save(image_path)

    # Define the mask area for the second girl to be placed next to the original
    width, height = image.size
    mask_area = (int(width * 0.5), 0, width, height)  # Right half of the image

    # Create a mask image to indicate where the second girl should be placed
    mask = Image.new("L", image.size, 0)  # Create a black image
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_area, fill=255)  # Create a white rectangle as the mask area

    # Save the mask
    mask_path = "temp_mask.png"
    mask.save(mask_path)

    # Prompt for generating a look-alike girl
    prompt = "two men in professional appearance"

    # Call OpenAI's inpainting API to generate the second girl and place her in the masked region
    try:
        edited_image_response = client.images.edit(
            image=open(image_path, "rb"),
            mask=open(mask_path, "rb"),
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="url"
        )
        edited_image_url = edited_image_response.data[0].url
        print(f"Edited image URL: {edited_image_url}")
        edited_image_response = requests.get(edited_image_url, verify=certifi.where())
        final_image = Image.open(BytesIO(edited_image_response.content))
    except Exception as e:
        print(f"Error during inpainting API call: {e}")
        return None

    # Cleanup temporary files
    try:
        os.remove(image_path)
        os.remove(mask_path)
        print("Temporary files deleted successfully.")
    except OSError as e:
        print(f"Error deleting temporary files: {e}")

    return final_image

# Load the uploaded image
# original_image_path = "/mnt/data/generated_image_2ae7a802-17ba-433a-9fc2-680b84e73a5b.png"
original_image_path = r"/openai_images_projects/images/generated_image_2ae7a802-17ba-433a-9fc2-680b84e73a5b.png"
original_image = Image.open(original_image_path)

# Apply the function to place the second girl (look-alike)
result_image = place_two_girls(original_image)

# Save or display the result image
if result_image:
    result_image.show()  # This will display the image on supported environments
    result_image.save("result_with_two_girls.png")  # Save the final image to disk
