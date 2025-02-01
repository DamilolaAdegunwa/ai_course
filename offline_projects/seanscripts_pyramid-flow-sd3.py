import os
import json
import torch
import numpy as np
import PIL
from PIL import Image
from IPython.display import HTML
from pyramid_dit import PyramidDiTForVideoGeneration
from IPython.display import Image as ipython_image
from diffusers.utils import load_image, export_to_video, export_to_gif

variant = 'diffusion_transformer_384p'  # For low resolution

model_path = "../pyramid-flow-sd3"  # The downloaded checkpoint dir
model_dtype = "bf16"

device_id = 0
torch.cuda.set_device(device_id)

model = PyramidDiTForVideoGeneration(
    model_path,
    model_dtype,
    model_variant=variant,
)

if model_dtype == "bf16":
    torch_dtype = torch.bfloat16
elif model_dtype == "fp16":
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32


def show_video(ori_path, rec_path, width="100%"):
    html = ''
    if ori_path is not None:
        html += f"""<video controls="" name="media" data-fullscreen-container="true" width="{width}">
        <source src="{ori_path}" type="video/mp4">
        </video>
        """

    html += f"""<video controls="" name="media" data-fullscreen-container="true" width="{width}">
    <source src="{rec_path}" type="video/mp4">
    </video>
    """
    return HTML(html)


prompt = "[prompt]"

output_name = "./text_to_video_sample.mp4"

# used for 384p model variant
width = 640
height = 384

temp = 16  # temp in [1, 31] <=> frame in [1, 241] <=> duration in [0, 10s]

torch.cuda.empty_cache()

model.vae.enable_tiling()

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True if model_dtype != 'fp32' else False, dtype=torch_dtype):
    frames = model.generate(
        prompt=prompt,
        num_inference_steps=[20, 20, 20],
        video_num_inference_steps=[10, 10, 10],
        height=height,
        width=width,
        temp=temp,
        guidance_scale=9.0,  # The guidance for the first frame
        video_guidance_scale=5.0,  # The guidance for the other video latent
        output_type="pil",
        save_memory=True,  # If you have enough GPU memory, set it to `False` to improve vae decoding speed
        cpu_offloading=True,  # Unload models after using them
    )

export_to_video(frames, output_name, fps=24)
show_video(None, output_name, "70%")