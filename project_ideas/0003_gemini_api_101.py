# !pip install -U -q "google-generativeai>=0.7.2" # Install the Python SDK

import google.generativeai as genai
import os
from apikey import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Give me python code to sort a list")
print(response.text)