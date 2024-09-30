# Exercise 1

from openai import OpenAI
from apikey import apikey
import os

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()

audio = open(r"C:\Users\damil\PycharmProjects\ai_course\resources\Erwin Smith's Words __ My Soldiers.mp4", "rb")

print("transcribing...")

response = client.audio.translations.create(
    model="whisper-1",
    file=audio,
    response_format="text"
)
print(response)