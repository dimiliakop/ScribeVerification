import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3PlusExactEmbedding(nn.Module):
    """
    Closer to paper:
      base features -> 7x7x576  (torchvision)
      1x1 reduce -> 7x7x160     (paper's 7x7x160)
      GAP -> 1x1x160
      1x1 -> 1x1x10             (paper's 10-D embedding)
    """
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = base.features
        self.reduce_160 = nn.Sequential(
            nn.Conv2d(576, 160, kernel_size=1, bias=False),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.to10 = nn.Conv2d(160, embedding_dim, kernel_size=1, bias=True)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.features(x)      # N x 576 x 7 x 7
        x = self.reduce_160(x)    # N x 160 x 7 x 7
        x = self.pool(x)          # N x 160 x 1 x 1
        x = self.to10(x)          # N x 10  x 1 x 1
        x = self.flat(x)          # N x 10
        return x

class SiameseNet(nn.Module):
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        self.embedding_net = MobileNetV3PlusExactEmbedding(embedding_dim, pretrained)

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2

def pairwise_l2(z1, z2):
    return torch.norm(z1 - z2, p=2, dim=1)
