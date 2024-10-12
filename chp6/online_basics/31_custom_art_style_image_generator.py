"""
Project: Fantasy Character Portrait Generator
In this project, we will create a script that generates unique fantasy character portraits based on descriptive prompts. The script will take in a list of character descriptions, generate images using the OpenAI API, and display or save the images. This project builds on your previous experience by adding complexity through the use of multiple character descriptions, allowing for more advanced creative outputs.

Python Code
"""
import os
from openai import OpenAI
from apikey import apikey  # Your file that stores the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_character_portrait(prompt):
    """
    Generate a fantasy character portrait based on a description prompt.

    :param prompt: A string containing the character description.
    :return: The generated image as a PIL Image object.
    """
    response = client.images.generate(
        prompt=prompt,
        size='1024x1024'
    )
    return response.data[0].url  # Returns the URL of the generated image


def create_character_portraits(prompts):
    """
    Create character portraits for a list of prompts.

    :param prompts: A list of strings containing character descriptions.
    :return: A list of generated image URLs.
    """
    portrait_urls = []

    for prompt in prompts:
        portrait_url = generate_character_portrait(prompt)
        portrait_urls.append(portrait_url)

    return portrait_urls


# Example use cases
if __name__ == "__main__":
    # List of character descriptions
    character_prompts = [
        "A brave knight wearing shining armor, holding a sword, standing in a forest.",
        "A mystical elf with long hair and green eyes, dressed in flowing robes, surrounded by magical creatures.",
        "A fierce dragon with emerald scales perched on a mountain peak.",
        "A cunning sorceress with a staff, casting a spell under a starry sky.",
        "A giant ogre with a club, living in a cave, with a mischievous grin."
    ]

    # Create character portraits
    portraits = create_character_portraits(character_prompts)

    # Display each generated character portrait URL
    for index, url in enumerate(portraits):
        print(f"Character Portrait {index + 1}: {url}")

"""
How to Test the Project:
Set Up Your Environment:

Ensure you have the required libraries installed. You can install them using pip:
bash
Copy code
pip install openai requests pillow
Configure API Key:

Create a file named apikey.py and store your OpenAI API key in it:
python
Copy code
apikey = 'YOUR_API_KEY_HERE'
Run the Script:

Execute the Python script. It will generate fantasy character portraits based on the provided descriptions and display the URLs of the generated images.
Example Use Cases:
Role-Playing Games (RPGs): Use the portraits to visualize characters in tabletop or online RPGs.
Character Design: Artists can use the generated portraits as inspiration for character designs in comics or video games.
Creative Writing: Writers can visualize their characters to enhance storytelling, making it easier to describe them in narratives.
Social Media Sharing: Share unique character portraits on social media platforms to engage followers with imaginative content.
This project provides an engaging way to utilize AI for creative character design, enhancing your skills while expanding your project repertoire.
"""