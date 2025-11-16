import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Embedding(nn.Module):
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        in_ch = net.fc.in_features
        net.fc = nn.Linear(in_ch, embedding_dim)  # GAP -> 10-D embedding
        self.model = net

    def forward(self, x):
        return self.model(x)  # N x 10

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        self.embedding_net = ResNet18Embedding(embedding_dim, pretrained)

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2

def pairwise_l2(z1, z2):
    return torch.norm(z1 - z2, p=2, dim=1)
