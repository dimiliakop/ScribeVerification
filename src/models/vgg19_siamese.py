import torch
import torch.nn as nn
import torchvision.models as models

class VGG19Embedding(nn.Module):
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.vgg19(weights=weights)

        # Replace the last classifier layer (1000 classes → embedding_dim)
        in_ch = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_ch, embedding_dim)

        self.model = net

    def forward(self, x):
        return self.model(x)

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        self.embedding_net = VGG19Embedding(embedding_dim, pretrained)

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2

def pairwise_l2(z1, z2):
    return torch.norm(z1 - z2, p=2, dim=1)
