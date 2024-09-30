"""
Project Title: AI-Powered Image Sequence Animation Generator
Description:
In this project, you will build an AI-powered image sequence animation generator that takes multiple text prompts and generates a sequence of images using OpenAI's DALL·E 3 API. The images will be treated as individual frames in an animation, and you will stitch them together into a GIF or video file. Each image will slightly differ from the previous one to simulate motion or progression based on a common theme (like an object morphing or moving through a scene).

This project focuses on image sequencing, time-based animations, and creating dynamic visuals based on AI-generated images.

Key Features:
Generate a sequence of images based on progressive prompts.
Create an animation (GIF or MP4) from the generated images.
Allow the user to specify the number of frames and animation speed (frames per second).
Handle API requests efficiently and create a smooth transition between frames.
Python Code:
"""

import os
import requests
from PIL import Image
from io import BytesIO
import certifi
import imageio
from openai import OpenAI
from apikey import apikey

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Set up the OpenAI API key
#openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a function to generate an image from a prompt
def generate_image(prompt):
    """
    Generate an image based on a text prompt using OpenAI's DALL·E 3 model.
    """
    response = client.images.generate(
        prompt=prompt,
        n=1,
        size="512x512",
        response_format="url"
    )
    image_url = response.data[0].url
    print(f"Image generated for prompt: '{prompt}' - URL: {image_url}")
    image_response = requests.get(image_url, verify=certifi.where())
    return Image.open(BytesIO(image_response.content))


# Define a function to generate a series of frames based on incremental prompts
def generate_image_sequence(base_prompt, steps=5):
    """
    Generate a sequence of images that incrementally evolve based on a base prompt.
    :param base_prompt: The base prompt for image generation.
    :param steps: Number of frames to generate.
    :return: A list of generated images.
    """
    images = []
    for step in range(steps):
        # Modify the base prompt slightly for each step to simulate progression
        prompt = f"{base_prompt}, scene step {step + 1} of {steps}"
        image = generate_image(prompt)
        images.append(image)
    return images


# Define a function to create an animation (GIF or MP4) from the sequence of images
def create_animation(images, output_file="animation.gif", fps=5):
    """
    Create an animation from a sequence of images and save it as a GIF or MP4.
    :param images: List of PIL images to animate.
    :param output_file: The filename for the output animation (GIF or MP4).
    :param fps: Frames per second for the animation speed.
    """
    if output_file.endswith(".gif"):
        # Create a GIF animation
        images[0].save(output_file, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)
    else:
        # Create an MP4 animation
        imageio.mimsave(output_file, [imageio.imread(img.filename) for img in images], fps=fps)
    print(f"Animation saved as {output_file}")


# Main function to generate and animate a sequence of images
def main():
    # Define the base prompt for the image sequence
    base_prompt = "A futuristic city skyline at sunset with flying cars"

    # Generate the sequence of images (e.g., 10 frames)
    image_sequence = generate_image_sequence(base_prompt, steps=10)

    # Save the image sequence as an animated GIF (or change to MP4)
    create_animation(image_sequence, output_file="city_animation.gif", fps=5)


if __name__ == "__main__":
    main()


"""
Key Learning Points:
Sequential Image Generation: You will learn how to modify prompts incrementally to create a smooth transition of images, simulating movement or evolution in a series of frames.
Creating Animations: You’ll get hands-on experience stitching individual images into a GIF or video format, handling both static and dynamic visuals.
Time-based Transitions: You’ll understand how to control the animation speed by adjusting the frames per second (fps), which directly impacts the smoothness of the animation.
Optimizing API Usage: You will practice handling multiple image requests efficiently while ensuring consistency in the style and progression of the images.
Explanation:
Image Sequence Generation: The core of the project lies in generating a series of images, where each image slightly changes based on a modified version of the original prompt. In this case, we progressively change the prompt by adding "scene step X of Y" to simulate different stages.
Prompt Progression: This allows you to create a visual evolution, such as a cityscape getting darker or a sunset changing colors as you move through the frames.
Animation Creation: Using imageio or PIL's save method, we combine these individual frames into a coherent animation. The animation speed (fps) controls how fast the sequence plays, and the output can either be a GIF or MP4.
Additional Challenges:
Advanced Prompt Evolution: Instead of simple step increments, modify the prompt with more dynamic changes, such as gradually changing the color scheme or altering objects within the scene.
User Input: Allow the user to input their own base prompt, the number of frames, and the animation type (GIF or MP4).
Smoother Transitions: Implement image interpolation or blending techniques to smooth out the transitions between generated frames, creating a more fluid animation.
Why It’s More Advanced:
Sequential Thinking: You’re not only generating individual images but now thinking of how they interact in sequence, which introduces complexity in planning and execution.
Animation Handling: Creating and controlling an animation introduces new concepts like frame rates and file formats (GIF/MP4), which are essential for time-based media creation.
Progressive Prompts: The incremental changes in prompts give the project a dynamic and evolving element, where slight changes between each frame can create stunning visual effects.
This project will push your understanding of both OpenAI's image generation capabilities and how to create cohesive, multi-frame animations from the generated images, opening the door to more complex video-based AI projects.
"""