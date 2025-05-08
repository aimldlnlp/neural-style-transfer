# utils/image_saver.py
import torch
from torchvision import transforms
from PIL import Image

import os

def save_image(tensor, path):
    tensor = tensor.clone().squeeze(0)
    tensor = tensor.cpu().clamp(0, 1)
    unloader = transforms.ToPILImage()
    image = unloader(tensor)
    # Check if path has a valid image extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    _, ext = os.path.splitext(path)
    if ext.lower() not in valid_extensions:
        path += '.png'  # default to png if no valid extension
    image.save(path)
