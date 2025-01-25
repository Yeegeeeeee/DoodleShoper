import base64

import requests
import logging

from conf.config import Config
from common.constants import *

url = Config.app_settings.get('sd_address')

def build_payload(prompt, base64_img):
    # A1111 payload
    return {
        "init_images": [base64_img],
        "prompt": prompt + ", high resolution, photorealistic, high detail, 8k uhd, dslr",
        "negative_prompt": "low resolution, cropped, person, text, out of frame, worst quality, low quality, centered, wide shot",
        "sampler_name": STABLE_DIFFUSION_SAMPLER,
        "steps": 20,
        "cfg_scale": 7,
        "denoising_strength": 0.75
    }

def generate_image_stable_diffusion(prompt, base64_img):
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Base64 Image Length: {len(base64_img) if base64_img else 0}")

        if not base64_img or len(base64_img) < 100:
            logging.error("Invalid base64 image data")
            return ""

        if base64_img.startswith('data:image'):
            base64_img = base64_img.split(',')[1]

        payload = build_payload(prompt, base64_img)

        try:
            response = requests.post(f'{url}{STABLE_DIFFUSION_PATH}', json=payload)
            logging.info(f"Response Status: {response.status_code}")
            logging.info(f"Response Body: {response.text[:500]}")

            if response.status_code != 200:
                logging.error(f"Error Details: {response.text}")
                return ""

            return response.json()['images'][0]

        except Exception as e:
            logging.error(f"Generation Error: {str(e)}")
            return ""