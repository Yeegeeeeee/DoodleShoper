import base64
import requests
import logging
from conf.config import Config
from common.constants import *

url = Config.app_settings.get('sd_address')


def build_payload(prompt, base64_img):
    if not base64_img:
        raise ValueError("base64_img cannot be empty")

    logging.debug(f"Payload base64 length: {len(base64_img)}")
    logging.debug(f"Payload base64 prefix: {base64_img[:50]}")

    if base64_img.startswith('data:image'):
        base64_img = base64_img.split(',')[1]

    payload = {
        "init_images": [base64_img],
        "prompt": prompt + ", high resolution, photorealistic, high detail, 8k uhd, dslr",
        "negative_prompt": "low resolution, cropped, person, text, out of frame, worst quality, low quality, centered, wide shot",
        "steps": 20,
        "cfg_scale": 7,
        "denoising_strength": 0.75
    }

    return payload


def generate_image_stable_diffusion(prompt, base64_img):
    try:
        logging.info(f"Generating image with prompt: {prompt}")
        logging.info(f"Base64 image length: {len(base64_img) if base64_img else 0}")

        if not base64_img or len(base64_img) < 100:
            raise ValueError("Invalid base64 image data")

        payload = build_payload(prompt, base64_img)

        api_url = f'{url.rstrip("/")}/{STABLE_DIFFUSION_PATH.lstrip("/")}'
        logging.info(f"Making request to: {api_url}")

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()
        if 'images' not in result or not result['images']:
            raise ValueError("No images in response")

        return result['images'][0]

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {str(e)}")
        logging.error(f"Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
        return ""
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Error: {str(e)}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected Error: {str(e)}")
        return ""