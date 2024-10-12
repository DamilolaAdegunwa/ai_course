import os
import requests
from PIL import Image, ImageFilter
from io import BytesIO
import numpy as np
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key
import torch
from torch import nn
from torchvision import transforms
import certifi
# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Super-Resolution model class (Basic example using a small CNN)
class SuperResolutionNet(nn.Module):
    def __init__(self):
        super(SuperResolutionNet, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# Initialize the super-resolution model
super_res_model = SuperResolutionNet()


# Image transformation functions
"""
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),  # Low-resolution input
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension
"""


def preprocess_image(image):
    """
    Preprocess the PIL image to convert it to a tensor suitable for the model.
    """
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
    ])

    return preprocess(image)

"""
def postprocess_image(tensor_image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),  # Upscale to high resolution
    ])
    return transform(tensor_image.squeeze())  # Remove batch dimension
"""

def postprocess_image(tensor):
    """
    Postprocess the output tensor from the model to convert it back to a PIL Image.
    """
    from torchvision import transforms

    # Remove batch dimension and clamp values to [0, 1] range
    tensor = tensor.squeeze(0).clamp(0, 1)

    # Convert the tensor back to a PIL image
    postprocess = transforms.ToPILImage()
    res = postprocess(tensor)
    return res

# Function to generate the image from the prompt using OpenAI’s API
def generate_image_from_prompt(prompt):
    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"
    )
    return response.data[0].url  # Return the URL of the generated image


# Function to download the generated image
def download_image(image_url):
    response = requests.get(image_url,verify=certifi.where())
    img = Image.open(BytesIO(response.content))
    return img


# Function to blur the image to simulate low resolution
def simulate_blur(image):
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))  # Apply slight blur
    return blurred_image


# Function to apply AI-based super-resolution
def apply_super_resolution(image, model):
    """
    Apply AI-based super-resolution on the given low-res image.
    :param image: PIL Image object (low resolution).
    :param model: Super-resolution model.
    :return: PIL Image object (high resolution).
    """
    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Ensure the model is in evaluation mode
    model.eval()

    # Use the model to predict the high-resolution image
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess the image to convert it back to PIL format
    high_res_image = postprocess_image(output_tensor)
    print("the type of high_res_image is: " + type(high_res_image))

    return high_res_image


# Function to save the image to a file
def save_image(image, filename):
    image.save(filename)
    print(f"Image saved as {filename}")


# Example use case
if __name__ == "__main__":
    # Step 1: Prompt to generate the base image
    prompt = "A medieval castle by a river surrounded by fog and dense forests"

    print(f"Generating image for prompt: '{prompt}'...")
    image_url = generate_image_from_prompt(prompt)

    # Step 2: Download the generated image
    base_image = download_image(image_url)
    #base_image.show()

    # Step 3: Simulate blur to represent a low-resolution image
    blurred_image = simulate_blur(base_image)
    #blurred_image.show()
    save_image(blurred_image, "images/blurred_castle_image.jpg")

    # Step 4: Apply AI-powered super-resolution to the blurred image
    print("Applying AI-based super-resolution to enhance details...")
    enhanced_image = apply_super_resolution(blurred_image, super_res_model)

    # Step 5: Save the super-resolution image
    save_image(enhanced_image, "images/super_res_castle_image.jpg")
    enhanced_image.show()