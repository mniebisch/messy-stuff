import pathlib

import numpy as np
import torch
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from diffusers.utils import load_image
from PIL import Image


def pad_to_multiple_of_8(
    pil_img: Image.Image, mode: str = "reflect"
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Pads a PIL image so both width and height are divisible by 8.
    Returns the padded image *and* the (left, top, right, bottom) padding tuple,
    so you can undo it after super-resolution.

    mode:
        "reflect"  – mirror the border pixels      (safest, avoids dark rims)
        "edge"     – repeat the outermost pixels
        "black"    – constant black padding
    """
    if mode not in {"reflect", "edge", "black"}:
        raise ValueError("mode must be 'reflect', 'edge' or 'black'")

    w, h = pil_img.size
    pad_r = (8 - w % 8) % 8  # right
    pad_b = (8 - h % 8) % 8  # bottom

    if pad_r == 0 and pad_b == 0:  # already /8 → nothing to do
        return pil_img, (0, 0, 0, 0)

    # convert to NumPy [H,W,C]
    arr = np.asarray(pil_img)

    if mode == "black":
        padded = np.pad(arr, ((0, pad_b), (0, pad_r), (0, 0)), constant_values=0)

    else:
        # build padding manually for reflect / edge
        if mode == "reflect":
            wrap = "reflect"
            # reflect does not repeat edge pixel, so handle size 1
            if w == 1:
                arr = np.concatenate([arr, arr], axis=1)
            if h == 1:
                arr = np.concatenate([arr, arr], axis=0)
        else:  # "edge"
            wrap = "edge"

        padded = np.pad(arr, ((0, pad_b), (0, pad_r), (0, 0)), mode=wrap)

    padded_img = Image.fromarray(padded)
    return padded_img, (0, 0, pad_r, pad_b)


dtype = torch.float16

output_dir = pathlib.Path("//mnt/data/fingerspelling5_upscaled")

input_filepath = pathlib.Path("/mnt/data/fingerspelling5/A/g/color_6_0002.png")

controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing()
pipe.vae.enable_tiling()  # call on VAE
# no xformers installed → Torch-2 SDPA is used automatically

scale_factor = 4

image = Image.open(input_filepath).convert("RGB")
width, height = image.size
image = image.resize((width * scale_factor, height * scale_factor))
image_padded, padding = pad_to_multiple_of_8(image, mode="reflect")


# control_image = load_image(
#     "https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler/resolve/main/examples/input.jpg"
# ).resize(
#     (1024, 1024)
# )  # 2× upscale test

with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    img = pipe(
        prompt="",
        control_image=image_padded,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=28,
        guidance_scale=3.5,
        height=image_padded.size[1],
        width=image_padded.size[0],
    ).images[0]

print(img.size)
img.save(output_dir / input_filepath.name)

# https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler
# https://huggingface.co/black-forest-labs/FLUX.1-dev
