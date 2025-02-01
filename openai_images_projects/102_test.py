import os

from openai.types import ImagesResponse

from apikey import apikey
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
import requests
import certifi
from io import BytesIO
from PIL import Image, ImageFile
import uuid
from openai import OpenAI
from httpx import URL, Proxy, Timeout, Response, ASGITransport, BaseTransport, AsyncBaseTransport

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

k: str = os.getenv("OPENAI_API_KEY")
print(f" the apikey is {k}")

client: OpenAI = OpenAI(api_key=apikey)
prompt: str = "johnny bravo waving to a crowd!"
print('generating image')

response: ImagesResponse = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    n=1
)
print(response)
disable_warnings(InsecureRequestWarning)
image_url: str = response.data[0].url
imageResponse: Response = requests.get(image_url, verify=certifi.where())  # get the image via httprequest
image_file: ImageFile = Image.open(BytesIO(imageResponse.content))
image_file.show()
guid: uuid.UUID = uuid.uuid4()
output_directory: str = f"images/{guid}/"
os.makedirs(output_directory, exist_ok=True)
filename: str = f"generated_image_{guid}.png"
print(f"the image filename is {filename}")
image_file.save(output_directory+filename)

