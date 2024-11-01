from PIL import Image
import config

def load_image(image_path):
    image = Image.open(image_path)
    image = config.loader(image).unsqueeze(0)
    return image.to(config.DEVICE)
