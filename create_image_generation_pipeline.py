### create image generation pipeline ###
# pip install setuptools - quick fix for distutils is removed in Python 3.12

import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
