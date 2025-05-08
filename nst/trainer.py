# nst/trainer.py
import torch
import torch.optim as optim
from torch.nn.functional import mse_loss
from models.vgg import VGGFeatures

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    return torch.mm(features, features.t()) / (c * h * w)

def train_style_transfer(content, style, device, steps=500, style_weight=1e6, content_weight=1):
    target = content.clone().requires_grad_(True).to(device)
    model = VGGFeatures().to(device)
    optimizer = optim.Adam([target], lr=0.003)

    content_features = model(content)
    style_features = model(style)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for i in range(steps):
        target_features = model(target)
        content_loss = mse_loss(target_features['conv4_2'], content_features['conv4_2'])

        style_loss = 0
        for layer in style_grams:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            style_loss += mse_loss(target_gram, style_gram)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Step {i}, Total Loss: {total_loss.item():.4f}")

    return target
