# utils/image_loader.py
from PIL import Image
import torchvision.transforms as transforms
import torch

def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape:
        size = shape
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = transform(image).unsqueeze(0)
    return image
