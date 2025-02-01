import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from online_module import *
from apikey import apikey
from PIL import Image, ImageDraw, ImageFont
from offline_module import *
import json


def wrap_text(text, font, max_width):
    """
    Wrap text to fit within a specified width when drawn with a specified font.
    """
    lines = []
    words = text.split()

    while words:
        line = ''
        while words and font.getmask(line + words[0]).getbbox()[2] <= max_width:
            line += (words.pop(0) + ' ')
        lines.append(line.strip())

    return lines


def text_on_image(image, text, font_path=None, position="top",
                  font_size=50, bg_color=(0, 0, 0, 128),
                  line_spacing=20, padding_top=20,
                  padding_bottom=20, padding_sides=0):
    draw = ImageDraw.Draw(image)

    # Load a font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Default font will be used as specified font was not found.")
        font = ImageFont.load_default()

    # Wrap text considering side padding
    max_width = image.width - 2 * padding_sides
    wrapped_text = wrap_text(text, font, max_width)

    # Calculate total text block height with line spacing
    text_heights = [font.getmask(line).getbbox()[3] for line in wrapped_text]
    text_block_height = sum(text_heights) + line_spacing * (len(wrapped_text) - 1)

    # Position the text at the top or bottom
    y = padding_top if position == "top" else image.height - text_block_height - padding_bottom

    # Draw background rectangle for text
    bg_rectangle_top = y - padding_top
    bg_rectangle_bottom = y + text_block_height + padding_bottom
    bg_rectangle_left = padding_sides
    bg_rectangle_right = image.width - padding_sides
    if position == "top":
        draw.rectangle([(bg_rectangle_left, bg_rectangle_top), (bg_rectangle_right, bg_rectangle_bottom)],
                       fill=bg_color)
    else:
        draw.rectangle([(bg_rectangle_left, image.height - text_block_height - padding_top - padding_bottom),
                        (bg_rectangle_right, image.height)], fill=bg_color)

    # Reset y position for text
    y = padding_top if position == "top" else image.height - text_block_height - padding_bottom

    # Draw text on image
    for line in wrapped_text:
        line_width = font.getmask(line).getbbox()[2]
        x = (image.width - line_width) / 2
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += font.getmask(line).getbbox()[3] + line_spacing

    # Display the image
    return image


#### Image Generation ####
st.title("Story Book Generator")
model_path_sdxl = ("../../Models/models--stabilityai--stable-diffusion-xl-base-1.0/"
                   "snapshots/462165984030d82259a11f4367a4eed129e94a7b")
model_path_sdxl_refiner = ("../../Models/models--stabilityai--stable-diffusion-xl-refiner-1.0/"
                           "snapshots/5d4cfe854c9a9a87939ff3653551c2b3c99a4356")

lora_path = ("../../Models/Loras/StorybookRedmondV2-KidsBook-KidsRedmAF.safetensors")

base, refiner = load_model_local_sdxl(model_path_sdxl, None, lora_path)

client = setup_openai(apikey)

user_input = st.text_input("Enter your prompt", value="A princess that saved the village from a dragon")
lora_trigger = "Kids Book"
prompt_user_input = user_input + lora_trigger

prompt = """
write a 5 page story for children with a character. 
- Each page will have a single line.
- For each page give a description that will be used to generate an image for that page. Don't use names in the description of the image. 
- Describe the appearance of the character in each image description, as the AI does not have a memory.
- Don't use words like the same character because the ai model does not have any memory. 
- Use white skin color for the character and mention it in each image description.
- follow the output format as follows:
{
"page1":"page 1 line",
"page2":"page 2 line",
"page3":"page 3 line",
"page4":"page 4 line",
"page5":"page 5 line",
"page1_image_description":"page1 image description",
"page2_image_description":"page2 image description",
"page3_image_description":"page3 image description",
"page4_image_description":"page4 image description",
"page5_image_description":"page5 image description",
}

Here are the details: 
"""

prompt = prompt + prompt_user_input

page1_text = "Once upon a time, in a faraway kingdom, there lived a brave princess."
page1_img = Image.open("img.jpg")

if st.button("Generate Story"):
    with st.spinner('Generating Story Book...'):

        story = generate_text_openai_streamlit(client, prompt)

        story_json = json.loads(story)

        for i in range(1, 6):
            st.text("Page " + str(i))
            st.write(story_json["page" + str(i)])
            st.write(story_json["page" + str(i) + "_image_description"])
            page_prompt = story_json["page" + str(i) + "_image_description"]
            page_text = story_json["page" + str(i)]
            generated_image = generate_image_local_sdxl(base, page_prompt)
            image = text_on_image(generated_image, page_text, font_path="one_trick_pony_tt.ttf",
                                  position='top')
            st.image(image, channels="BGR", use_column_width=True)