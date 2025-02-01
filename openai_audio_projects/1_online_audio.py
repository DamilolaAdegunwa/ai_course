# Exercise 1 - https://chatgpt.com/c/66fa9dcf-8df4-800c-a2ce-5b4b40c5d532

from openai import OpenAI
from apikey import apikey, filepath
import os

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()

audio = open(r"/resources/Erwin Smith's Words __ My Soldiers.mp4", "rb")

print("transcribing...")

response = client.audio.translations.create(
    model="whisper-1",
    file=audio,
    response_format="text"
)
print(response)