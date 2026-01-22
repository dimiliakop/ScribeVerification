import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViT_Embedding(nn.Module):
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        return self.vit(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        self.embedding_net = ViT_Embedding(embedding_dim, pretrained)

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2


def pairwise_l2(z1, z2):
    return torch.norm(z1 - z2, p=2, dim=1)
