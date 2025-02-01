"""
Project Title: Generative Art Based on Emotion and Artistic Styles
In this advanced exercise, you will create generative art by combining emotional themes with specific artistic styles (e.g., surrealism, impressionism, abstract). The focus of this project is to generate images that reflect human emotions while also applying distinct art styles to the output, resulting in highly expressive and visually varied imagery. The project involves dynamically combining emotional states and artistic styles to produce unique and imaginative outputs.

Python Code
"""
import os
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Function to generate images based on emotional themes and artistic styles
def generate_emotion_art(emotion, art_style):
    """
    Generate an image based on a specific emotional theme and artistic style.

    :param emotion: The emotional theme of the image (e.g., 'happiness', 'melancholy', 'fear').
    :param art_style: The artistic style to apply (e.g., 'surrealism', 'impressionism', 'abstract').
    :return: URL of the generated image.
    """
    prompt = f"A {art_style} painting that expresses {emotion}"

    response = client.images.generate(
        prompt=prompt,
        size="1024x1024"
    )

    return response.data[0].url  # Returns the URL of the generated image

def download_image(image_url):
    """
    Download an image from a URL and return it as a PIL Image object.

    :param image_url: The URL of the image.
    :return: PIL Image object
    """
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def save_image(image, filename):
    """
    Save the image to the specified file.

    :param image: PIL Image object.
    :param filename: The path and name of the file to save the image to.
    """
    image.save(filename)
    print(f"Image saved as {filename}")

# Example use cases
if __name__ == "__main__":
    # Define a set of emotional themes and artistic styles
    emotion_style_set = [
        ("happiness", "impressionism"),
        ("melancholy", "surrealism"),
        ("fear", "abstract"),
        ("serenity", "watercolor"),
        ("excitement", "pop art")
    ]

    # Loop through each combination and generate images
    for i, (emotion, style) in enumerate(emotion_style_set):
        print(f"Generating art for emotion '{emotion}' in style '{style}'...")
        image_url = generate_emotion_art(emotion, style)
        image = download_image(image_url)
        save_image(image, f"emotion_art_{i + 1}_{emotion}_{style}.jpg")
"""
Multiple Example Inputs and Expected Outputs
Input:

Emotion: "happiness"
Art Style: "impressionism"
Expected Output:
A bright and lively impressionist painting with soft brushstrokes, featuring vivid colors that express joy and contentment.
Input:

Emotion: "melancholy"
Art Style: "surrealism"
Expected Output:
A surrealist artwork showing a somber, dreamlike world with distorted shapes and muted colors, evoking a sense of sadness and reflection.
Input:

Emotion: "fear"
Art Style: "abstract"
Expected Output:
An abstract piece with chaotic, jagged lines, dark shades, and unsettling shapes, symbolizing fear and anxiety.
Input:

Emotion: "serenity"
Art Style: "watercolor"
Expected Output:
A peaceful and calming watercolor painting, using gentle washes of color to depict tranquil landscapes and evoke a sense of calm and stillness.
Input:

Emotion: "excitement"
Art Style: "pop art"
Expected Output:
A vibrant and energetic pop art piece, featuring bold colors, sharp contrasts, and lively patterns that convey a feeling of thrill and exuberance.
Project Overview
This project introduces the concept of combining human emotions with different artistic styles, pushing your image generation skills into the realm of generative art. By experimenting with various emotions and art movements, you can explore how context and visual style can influence the mood and tone of your generated images.
"""