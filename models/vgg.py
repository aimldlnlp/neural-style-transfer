# models/vgg.py
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import torchvision.models as models

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        # vgg = models.vgg19(pretrained=True).features[:21].eval()
        # vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
        # vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:30].eval()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.model = vgg

    def forward(self, x):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                  '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features
