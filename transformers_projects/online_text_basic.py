from openai import OpenAI
from apikey import apikey
import os

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()
prompt = "Write an essay on computer vision"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}])

print("Response")
print(response)
print("Message Content:")
print(response.choices[0].message.content)