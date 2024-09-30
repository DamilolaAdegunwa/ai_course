# Exercise 3

#import the necessary packages
import certifi
from io import BytesIO
import requests
#from requests.packages.urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
from PIL import Image, ImageEnhance
from openai import OpenAI
from apikey import apikey
import os
import uuid

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()
prompt = "A photorealistic portrait of a 20-year-old Japanese woman with luscious brown curly hair and striking blue eyes, wearing a tailored formal dress, seated gracefully at a polished wooden table in a stylish café, full body shot, 8k hdr, high detailed, lot of details, high quality, she has an expressive and inviting smile, bathed in warm, soft afternoon sunlight filtering through the window, surrounded by tasteful décor."

print('generating image')

response = client.images.generate(
    model="dall-e-3", # model="dall-e-2"
    prompt=prompt,
    size="1024x1024",
    n=1
)

print("Response")
print(response)

#to ignore 'InsecureRequestWarning'
#requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
disable_warnings(InsecureRequestWarning)

#get the image url from the data/object/json
image_url = response.data[0].url

#get the image. you can use this to get any image into python if you have the url
image = requests.get(image_url, verify=certifi.where())
#image = requests.get(image_url, verify=False)

#this opened in my photos app
image = Image.open(BytesIO(image.content))
image.show()

# Generate a random UUID for the image file
guid = uuid.uuid4()

#this could be considered save as
# Create the directory for the images
output_directory = f"images/{guid}/"
os.makedirs(output_directory, exist_ok=True)
filename = f"generated_image_{guid}.png"
image.save(output_directory+filename)

enhancer = ImageEnhance.Brightness(image)
img = enhancer.enhance(0.3)
img.show()
print('done!!')