import os
import certifi
import requests
from openai import OpenAI
from apikey import apikey  # Store your OpenAI key in apikey.py
from fastapi import FastAPI

app = FastAPI()

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()

@app.post("/generate_image/")
def generate_image(prompt: str):
    print(f"the prompt from generate_image is: {str}")
    response = client.images.generate(prompt=prompt, n=1, size="1024x1024")
    image_url = response.data[0].url
    img_response = requests.get(image_url, verify=certifi.where())
    return img_response.content  # Return image bytes for integration

# uvicorn image_generation_service:app --reload --port 8001
# http://localhost:8001/docs