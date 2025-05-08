# Exercise 1
from openai import OpenAI
from apikey import apikey
import os

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()
prompt = "Which city is the largest in the world by landmass? also tell me what is the number of people there"


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    # model="gpt-3.5-turbo-0125",
    messages=[{"role": "user", "content": prompt}]
)

print("Response")
print(response.choices[0].message.content)
