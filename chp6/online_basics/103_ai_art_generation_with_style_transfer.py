import os
from openai import OpenAI
from apikey import apikey  # Your apikey.py file that stores the key
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models import vgg19
from torch import nn
import certifi

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


# Function to generate image using OpenAI's image generation
def generate_image_from_prompt(prompt):
    response = client.images.generate(prompt=prompt, n=1, size='1024x1024')
    image_url = response.data[0].url
    return image_url


# Function for style transfer
def load_vgg_model():
    model = vgg19(pretrained=True).features
    for param in model.parameters():
        param.requires_grad = False
    return model


def apply_style_transfer(content_img, style_img, model, num_steps=300, style_weight=1e6, content_weight=1e0):
    content_img = preprocess_image(content_img).unsqueeze(0)
    style_img = preprocess_image(style_img).unsqueeze(0)

    input_img = content_img.clone()
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    style_loss_layers, content_loss_layers = get_layers_for_loss(model, style_img, content_img)

    for i in range(num_steps):
        def closure():
            optimizer.zero_grad()
            model(input_img)
            style_loss, content_loss = compute_loss(model, style_loss_layers, content_loss_layers)
            total_loss = style_weight * style_loss + content_weight * content_loss
            total_loss.backward()
            return total_loss

        optimizer.step(closure)

    output_img = postprocess_image(input_img)
    return output_img


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    return preprocess(image)


def postprocess_image(tensor):
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(1 / 255)),
        transforms.ToPILImage()
    ])
    image = tensor.clone().squeeze(0)
    return postprocess(image)


def get_layers_for_loss(model, style_img, content_img):
    # Define layers for style and content loss calculation
    pass


def compute_loss(model, style_loss_layers, content_loss_layers):
    # Implement content and style loss calculations
    pass


# Streamlit web app interface
st.title("AI Art Generation with Style Transfer")
st.write("Generate an AI image and apply a specific artistic style using neural networks.")

# User input for the text prompt
prompt = st.text_input("Enter a prompt to generate an image:", value="A futuristic city at sunset")

# Button to trigger image generation
if st.button("Generate Image"):
    st.write("Generating image...")
    image_url = generate_image_from_prompt(prompt)
    response = requests.get(image_url, verify=certifi.where())
    generated_image = Image.open(BytesIO(response.content))

    st.image(generated_image, caption="Generated Image", use_column_width=True)

    # Style selection
    style_option = st.selectbox("Select an artistic style", [
    "Impressionist", "Cubist", "Surrealist", "Realism", "Pop Art", "Abstract Expressionism",
    "Futurism", "Minimalism", "Baroque", "Romanticism", "Art Nouveau", "Dadaism", "Symbolism",
    "Neoclassicism", "Constructivism", "Renaissance", "Conceptual Art", "Post-Impressionism",
    "Expressionism", "Photorealism", "Op Art", "Bauhaus", "Street Art", "Suprematism",
    "Naïve Art", "Fauvism", "Hyperrealism", "Vorticism", "Rococo", "Lyrical Abstraction",
    "Precisionism", "De Stijl", "Tachisme", "Neo-Expressionism", "Art Deco"
])

    # Load style image (in practice, you'd have some predefined style images)
    style_image = Image.open(f"art_styles/{style_option.lower().replace(" ", "_")}.png")  # Assume saved style images for each

    # Apply style transfer
    if st.button("Apply Style Transfer"):
        model = load_vgg_model()
        styled_image = apply_style_transfer(generated_image, style_image, model)
        st.image(styled_image, caption="Styled Image", use_column_width=True)
