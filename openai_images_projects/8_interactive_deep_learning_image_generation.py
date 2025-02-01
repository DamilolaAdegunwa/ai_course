# 8 https://chatgpt.com/c/66f60116-7f44-800c-9d44-f39ed9d91833 - Project: Interactive Deep Learning Image Generation with Super-Resolution Enhancement

#import openai
import requests
import certifi
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFile
import os
import uuid
import random
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg19
from openai import OpenAI
from apikey import apikey  # assuming you have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Set OpenAI API key
#openai.api_key = os.getenv("OPENAI_API_KEY")

# Artistic styles for prompt variation
art_styles = [
    "Impressionist",
    "Cubist",
    "Surrealist",
    "Realism",
    "Pop Art",
    "Abstract Expressionism",
    "Futurism",
    "Minimalism",
    "Baroque",
    "Romanticism",
    "Art Nouveau",
    "Dadaism",
    "Symbolism",
    "Neoclassicism",
    "Constructivism",
    "Renaissance",
    "Conceptual Art",
    "Post-Impressionism",
    "Expressionism",
    "Photorealism"
]

# Define the base prompt
base_prompt = "A highly detailed portrait of a {subject}, in the {style} style."

# Function to save image in a generated directory
def create_save_directory():
    directory = f"images/{uuid.uuid4()}"
    os.makedirs(directory, exist_ok=True)
    return directory

# Create save directory
save_directory = create_save_directory()

# Function to generate a dynamic prompt
def generate_prompt(subject):
    style = random.choice(art_styles)
    return base_prompt.format(subject=subject, style=style), style

# Image generation function using OpenAI's API
def generate_image(prompt):
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024",
        n=1
    )
    image_url = response.data[0].url



    # Save the enhanced image
    guid = uuid.uuid4()
    image_filename = f"{style}_original_{guid}.png"
    image_path = os.path.join(save_directory, image_filename)

    print('here is the image url: ' + image_url)
    image_response = requests.get(image_url, verify=certifi.where())
    image_content = Image.open(BytesIO(image_response.content))
    image_content.save(image_path)
    return image_content, guid

# Super-resolution model (ESRGAN) - simplified placeholder
class SuperResolutionNet_Old(nn.Module):
    def __init__(self):
        super(SuperResolutionNet_Old, self).__init__()
        self.model = vgg19(pretrained=True).features[:35]  # Using part of VGG19 for simplicity

    def forward(self, x):
        return self.model(x)

# Apply super-resolution to the image
def apply_super_resolution_Old(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resizing the image for the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Simulate super-resolution by passing through a pre-trained model
    model = SuperResolutionNet_Old().eval()
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_image = transforms.ToPILImage()(output_tensor.squeeze())
    return output_image

# User-controlled image adjustment (brightness, contrast, sharpness)
def adjust_image(img: ImageFile) -> ImageFile:
    print("Would you like to adjust the image's brightness, contrast, or sharpness?")
    options = input("Enter 'brightness', 'contrast', 'sharpness', or 'none': ").lower()

    if options == 'brightness':
        print('dumb test')
        img.show('testing')
        factor = float(input("Enter a brightness factor (hint: just give between 0.0 to 1.0): "))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        img.show('testing-afterwards')
    elif options == 'contrast':
        factor = float(input("Enter a contrast factor (hint: just give between 0.0 to 1.0): "))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)
    elif options == 'sharpness':
        factor = float(input("Enter a sharpness factor (hint: just give between 0.0 to 2.0): "))
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)

    return img

# ---------------------------------------------------------------------------
# added this code for correction

# Super-resolution model (VGG19 feature extraction + dimensionality reduction)
class SuperResolutionNet(nn.Module):
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        self.model = vgg19(weights='IMAGENET1K_V1').features[:35]  # Using part of VGG19 for simplicity
        # Additional layers to convert the 512 channels to 3 (RGB)
        self.conv1x1 = nn.Conv2d(512, 3, kernel_size=1)  # Reducing to 3 channels (RGB)

    def forward(self, x):
        x = self.model(x)
        x = self.conv1x1(x)  # Apply 1x1 convolution to reduce channels to 3
        return x

# Apply super-resolution to the image
def apply_super_resolution(image) -> ImageFile:
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resizing the image for the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Simulate super-resolution by passing through a pre-trained model
    model = SuperResolutionNet().eval()
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Apply inverse normalization
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
    ])

    # Convert tensor back to PIL image
    pil_image = inv_transform(output_tensor.squeeze())

    # Save PIL image to a BytesIO buffer to simulate saving to a file
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)  # Reset buffer position to the start

    # Load the image from buffer as ImageFile
    image_file = ImageFile.Image.open(buffer)

    return image_file


# ---------------------------------------------------------------------------
# Main project flow
if __name__ == "__main__":
    subject = input("Enter the subject of your image (e.g., cat, landscape, person): ")

    ## Create save directory
    #save_directory = create_save_directory()

    # Generate and enhance multiple images
    generated_images = []
    for _ in range(1):
        # Generate a prompt and corresponding image
        prompt, style = generate_prompt(subject)
        print(f"Generating image with prompt: '{prompt}'")
        print('show the original')
        image, guid = generate_image(prompt)
        image.show("show_the_original")

        # Apply super-resolution
        # enhanced_image: ImageFile = apply_super_resolution(image)

        # Allow user to interactively adjust the image
        enhanced_image = adjust_image(image)

        # Save the enhanced image
        image_filename = f"{style}_enhanced_{guid}.png"
        image_path = os.path.join(save_directory, image_filename)
        enhanced_image.show()
        enhanced_image.save(image_path)
        print(f"Image saved as {image_path}")

        generated_images.append(enhanced_image)

    print(f"All images saved in directory: {save_directory}")
