# main.py
import torch
from utils.image_loader import load_image
from utils.image_saver import save_image
from nst.trainer import train_style_transfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_path = "data/content1.jpg"
style_path = "data/style1.jpg"
output_path = "output/result.jpg"

content = load_image(content_path).to(device)
style = load_image(style_path, shape=content.shape[-2:]).to(device)

output = train_style_transfer(content, style, device)
save_image(output, output_path)
