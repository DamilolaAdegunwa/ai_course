import os

import ctransformers
import streamlit as st
import torch
from diffusers import DiffusionPipeline, AutoPipelineForInpainting
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers import pipeline
from threading import Thread


############################################
# Inpaint
############################################
@st.cache_resource()
def load_model_inpaint(model_path_or_name, torch_dtype=torch.float16):
    model = AutoPipelineForInpainting.from_pretrained(model_path_or_name, torch_dtype=torch_dtype)
    model.enable_model_cpu_offload()
    return model


############################################
# Pipeline
############################################
@st.cache_resource()
def load_model_pipeline(task, model_path, device="cuda"):
    model = pipeline(task, model=model_path, device=device)
    return model


############################################
# Local Audio to Text (Whisper)
############################################
@st.cache_resource()
def load_model_audio(model_path, chunk_length_s=30, device="cuda"):
    model = pipeline("automatic-speech-recognition",
                     model=model_path, chunk_length_s=chunk_length_s, device=device)
    return model


def save_uploaded_audio_file(uploaded_file):
    try:
        # Create a directory to save the file
        os.makedirs("tempDir", exist_ok=True)
        file_path = os.path.join("tempDir", uploaded_file.name)

        # Write the file to the directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        # Handle exceptions
        return None


def generate_text_from_audio_local(model, audio_file):
    file_path = save_uploaded_audio_file(audio_file)

    result = model(file_path)["text"]
    return result


############################################
# Image Generation SDXL Model
############################################
@st.cache_resource()
def load_model_local_sdxl(model_path_sdxl, model_path_sdxl_refiner=None, model_path_lora=None,
                          torch_dtype=torch.float16,
                          variant="fp16", use_safetensors=True):
    #### BASE MODEL ####
    base = DiffusionPipeline.from_pretrained(model_path_sdxl,
                                             torch_dtype=torch_dtype,
                                             variant=variant,
                                             use_safetensors=use_safetensors)
    if model_path_lora:
        base.load_lora_weights(model_path_lora)

    base.enable_model_cpu_offload()

    #### REFINER MODEL ####
    if model_path_sdxl_refiner:
        refiner = DiffusionPipeline.from_pretrained(
            model_path_sdxl_refiner,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            variant=variant,
        )
        refiner.enable_model_cpu_offload()
    else:
        refiner = None

    return base, refiner


def generate_image_local_sdxl(base, prompt, refiner=None, num_inference_steps=20, guidance_scale=15,
                              high_noise_frac=0.8, output_type="latent", verbose=False, temprature=0.7):
    if refiner:
        image = base(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=high_noise_frac,
                     output_type=output_type, verbose=verbose, guidance_scale=guidance_scale,
                     temprature=temprature).images
        image = refiner(prompt=prompt, num_inference_steps=num_inference_steps, denoising_start=high_noise_frac,
                        image=image, verbose=verbose).images[0]
    else:

        image = base(prompt=prompt, num_inference_steps=num_inference_steps,
                     guidance_scale=guidance_scale).images[0]
    return image


############################################
# Text Generation using Lama2 GGUF Model
############################################

@st.cache_resource()
def load_model_text_llama2_gguf(model_path, model_file, model_type="llama", gpu_layers=50):
    model = ctransformers.AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=model_path,
                                                               model_file=model_file,
                                                               model_type=model_type, gpu_layers=gpu_layers)
    return model


def generate_text_llama2_gguf(model, prompt, text_area_placeholder, stop=["\n", "Question:", "Q:"]):
    generated_text = ""
    for text in model(f"Question: {prompt} Answer:",
                      stream=True, stop=stop):
        generated_text += text
        text_area_placeholder.markdown(generated_text, unsafe_allow_html=True)
    return generated_text


############################################
# Text Generation using Mistral AI Model
############################################

@st.cache_resource()
def load_model_text_local(model_path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


# Function to generate text
def generate_text_local_streamlit(model, tokenizer, prompt, text_area_placeholder,
                                  max_new_tokens=100, do_sample=True, top_p=0.95, top_k=1000, temperature=1.0,
                                  timeout=10.0, device="cuda"):
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
    input_ids = encoded_input.to(device)

    # Set up TextStreamer for streaming output
    streamer = TextIteratorStreamer(tokenizer, timeout=timeout, skip_prompt=True, skip_special_tokens=True)

    generate_args = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "streamer": streamer,
        "do_sample": do_sample,
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        # "num_beams" : 1,
        # "stopping_criteria" : StoppingCriteriaList([stop])
        "pad_token_id": tokenizer.eos_token_id  # Set pad_token_id to eos_token_id

    }
    thread = Thread(target=model.generate, kwargs=generate_args)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text

        text_area_placeholder.markdown(generated_text, unsafe_allow_html=True)

    return generated_text


def main():
    # ####### AUDIO TRANSCRIPTION #######
    # st.title("Audio Transcription using OpenAI API")
    # model_path = ("../Models/models--openai--whisper-medium/"
    #               "snapshots/18530d7c5ce1083f21426064b85fbd1e24bd1858")
    # model = load_model_audio(model_path)
    # audio_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav"])
    #
    # if audio_file:
    #     if st.button("Transcribe"):
    #         st.audio(audio_file, format='audio/wav')
    #         with st.spinner('Transcribing audio...'):
    #             result = generate_text_from_audio_local(model,audio_file)
    #             st.write(result)

    # #### Image Generation ####
    # st.title("Image Generation using Local SDXL Model")
    # model_path_sdxl = ("../Models/models--stabilityai--stable-diffusion-xl-base-1.0/"
    #                    "snapshots/462165984030d82259a11f4367a4eed129e94a7b")
    # model_path_sdxl_refiner = ("../Models/models--stabilityai--stable-diffusion-xl-refiner-1.0/"
    #                            "snapshots/5d4cfe854c9a9a87939ff3653551c2b3c99a4356")
    #
    # lora_path = ("../Models/models--artificialguybr--ColoringBookRedmond-V2/"
    #              "snapshots/c486ec1307a5dbf41c743392559ead0c82bdb614")
    # base,refiner = load_model_local_sdxl(model_path_sdxl,model_path_sdxl_refiner,lora_path)
    # prompt = st.text_input("Enter your prompt", value="lion")
    # lora_trigger = ",ColoringBookAF, Coloring Book"
    # if st.button("Generate Image"):
    #     with st.spinner('Generating image...'):
    #         image = generate_image_local_sdxl(base, prompt+lora_trigger,refiner)
    #         st.image(image)

    # # ##### Text generation Faster ####
    # st.title("Text Generation using Lama2 GGUF Model")
    #
    # model_path = "../Models/models--TheBloke--Llama-2-7B-GGUF/"\
    #          "snapshots/b4e04e128f421c93a5f1e34ac4d7ca9b0af47b80"
    # model_file = "llama-2-7b.Q4_K_M.gguf"
    # model = load_model_text_llama2_gguf(model_path,model_file)
    # prompt = st.text_input("Enter your prompt", value="Write an essay on computer vision")
    # text_area_placeholder = st.empty()
    # if st.button("Generate Text"):
    #     result = generate_text_llama2_gguf(model, prompt, text_area_placeholder)

    # ##### Text generation ####
    st.title("Text Generation using Mistral AI Model")

    model_path = ("../Models/models--mistralai--Mistral-7B-Instruct-v0.2/"
                  "snapshots/b70aa86578567ba3301b21c8a27bea4e8f6d6d61")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_text_local(model_path, device)
    prompt = st.text_input("Enter your prompt", value="Write an essay on computer vision")
    text_area_placeholder = st.empty()
    if st.button("Generate Text"):
        result = generate_text_local_streamlit(model, tokenizer, prompt, text_area_placeholder, device=device)


if __name__ == '__main__':
    main()