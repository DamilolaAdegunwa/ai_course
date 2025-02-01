"""
Project Title: Dynamic Landscape Generator Based on Seasons and Time of Day
File Name: dynamic_landscape_generator.py

Project Description: This project will dynamically generate landscape images based on two key parameters: the season and the time of day. By inputting the season (e.g., winter, summer) and time of day (e.g., morning, night), you will create themed landscape images such as a snowy night or a bright summer morning. The project builds on your previous work but introduces more complexity by incorporating different themes and blending multiple concepts into a single prompt.

The code will generate a descriptive image based on combinations of seasons and times of day, allowing you to explore how prompt variations impact the generated results.

Python Code:
"""
import os
from openai import OpenAI
from apikey import apikey  # The file containing the API key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

# Function to generate landscape images based on season and time of day
def generate_landscape(season, time_of_day):
    prompt = f"A breathtaking {season} landscape during the {time_of_day}, highly detailed, featuring nature elements and a serene atmosphere."
    size = "1024x1024"  # Image size defined explicitly
    try:
        response = client.images.generate(
            prompt=prompt,
            size=size
        )
        return response.data[0].url
    except Exception as e:
        return f"An error occurred: {e}"

# Example test cases
if __name__ == "__main__":
    # List of test cases for easy testing
    test_cases = [
        ("winter", "night"),
        ("summer", "morning"),
        ("spring", "afternoon"),
        ("autumn", "evening"),
        ("winter", "dawn")
    ]

    # Generate and print the URLs of the images for each test case
    for season, time_of_day in test_cases:
        image_url = generate_landscape(season, time_of_day)
        print(f"Generated for {season} at {time_of_day}: {image_url}")
"""
Example Inputs and Expected Outputs:
Input:

Season: winter
Time of Day: night
Expected Output:

An image of a snow-covered landscape under the moonlight, with stars visible in the dark sky. The image should evoke a calm winter night.
Input:

Season: summer
Time of Day: morning
Expected Output:

A bright, sunny landscape featuring clear skies, lush green trees, and flowers in full bloom, representing a peaceful summer morning.
Input:

Season: spring
Time of Day: afternoon
Expected Output:

A vivid image of blooming flowers, lush green fields, and a sun-drenched landscape, symbolizing the liveliness of spring in the afternoon.
Input:

Season: autumn
Time of Day: evening
Expected Output:

A landscape with vibrant autumn leaves and soft, warm lighting from the setting sun, giving a peaceful evening glow.
Input:

Season: winter
Time of Day: dawn
Expected Output:

A crisp, cold dawn with snow-covered trees and a gentle sunrise illuminating the white landscape with hues of pink and orange.
This project allows you to explore how different settings influence the mood and details of a generated image. By focusing on combinations of seasons and times of day, you’ll expand your ability to craft specific, nuanced prompts for generating thematic art.
"""