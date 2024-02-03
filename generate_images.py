### Image Generation ###
import os
import random

import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline

# Load pre-trained model
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

# Create directories to save generated images based on emotions
os.makedirs('/content/faces/happy', exist_ok=True)
os.makedirs('/content/faces/sad', exist_ok=True)
os.makedirs('/content/faces/angry', exist_ok=True)
os.makedirs('/content/faces/surprised', exist_ok=True)

# Define ethnicity and gender options
ethnicities = ['a latino', 'a white', 'a black', 'a middle eastern', 'an indian', 'an asian']
genders = ['male', 'female']

# Define emotion prompts
emotion_prompts = {'happy': 'smiling',
                   'sad': 'frowning, sad face expression, crying',
                   'surprised': 'surprised, opened mouth, raised eyebrows',
                   'angry': 'angry'}

# Generate 250 images for each emotion
for j in range(250):
    # Iterate through each emotion
    for emotion in emotion_prompts.keys():
        emotion_prompt = emotion_prompts[emotion]

        # Randomly choose ethnicity and gender
        ethnicity = random.choice(ethnicities)
        gender = random.choice(genders)

        # Define the prompt for image generation
        prompt = 'Medium-shot portrait of {} {}, {}, front view, looking at the camera, color photography, '.format(
            ethnicity, gender, emotion_prompt) + \
                 'photorealistic, hyperrealistic, realistic, incredibly detailed, crisp focus, digital art, depth of field, 50mm, 8k'

        # Define negative prompt for image generation
        negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ' + \
                          '((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, bad anatomy, blurred, watermark, grainy, signature'

        # Generate image using the Diffusion model
        img = pipeline(prompt, negative_prompt=negative_prompt).images[0]

        # Save generated image with appropriate filename
        img.save('/content/faces/{}/{}.png'.format(emotion, str(j).zfill(4)))

        # Uncomment the following lines if you want to display the images
        # plt.imshow(img)
        # plt.show()
