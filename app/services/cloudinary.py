import cloudinary
import cloudinary.uploader
import logging

from conf.config import Config

def init_cloudinary():
    cloudinary.config( 
        cloud_name = Config.app_settings.get('cloudinary_name'), 
        api_key = Config.app_settings.get('cloudinary_key'), 
        api_secret = Config.app_settings.get('cloudinary_secret')
    )

def upload_image_web(img_base64, img_uuid):
    try:
        if not img_base64:
            logging.error("Image base64 data is empty")
            return None

        cleaned_base64 = img_base64.strip() if img_base64 else ""

        result = cloudinary.uploader.upload(
            f"data:image/png;base64,{cleaned_base64}",
            public_id=img_uuid
        )
        img_url = result['url']
        logging.info(f"Uploaded image to cloudinary: {img_uuid} -> {img_url}")
        return img_url
    except Exception as e:
        logging.error(f"Upload failed: {str(e)}")
        return None

def destroy_image_web(img_uuid):
    cloudinary.uploader.destroy(img_uuid)
    logging.info("Destroyed image from cloudinary with uuid: {}".format(img_uuid),)