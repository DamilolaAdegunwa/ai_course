# Exercise 3

# import the necessary packages
import certifi  # to solve issues with ssl
from io import BytesIO  # to convert to and from bytes
import requests  # to make httpRequest and get the json/payload from an api
# from requests.packages.urllib3.exceptions import InsecureRequestWarning # to try resolve the issue with ssl
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
from PIL import Image, ImageEnhance
from openai import OpenAI
from apikey import apikey  # custom page for keeping the apikey and other stuffs
import os  # for storing config
import uuid  # for creating guid/uuid

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()
prompt = ("A photorealistic portrait of a 20-year-old Japanese woman with luscious brown curly hair and striking blue "
          "eyes, wearing a tailored formal dress, seated gracefully at a polished wooden table in a stylish café, "
          "full body shot, 8k hdr, high detailed, lot of details, high quality, she has an expressive and inviting "
          "smile, bathed in warm, soft afternoon sunlight filtering through the window, surrounded by tasteful décor.")

print('generating image')

response = client.images.generate(
    model="dall-e-3",  # model="dall-e-2"
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

# Generate a random UUID for the image file
guid = uuid.uuid4()

#this could be considered save as
# Create the directory for the images
output_directory = f"images/{guid}/"
os.makedirs(output_directory, exist_ok=True)  # basically, if the dir doesn't exist, create it
filename = f"generated_image_{guid}.png"
image.save(output_directory+filename)
image.show(title=filename)
print(f"we save the original image to {output_directory+filename}")

enhancer = ImageEnhance.Brightness(image)
br_img = enhancer.enhance(0.3)
br_filename = f"generated_image_brightness_{guid}.png"
br_img.save(output_directory+br_filename)
br_img.show(title=br_filename)
print(f"we save the brighter image to {output_directory+br_filename}")
print('done!!')