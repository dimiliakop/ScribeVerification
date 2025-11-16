import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3PlusEmbedding(nn.Module):
    """
    Feature extractor for scribe verification:
    - Based on MobileNetV3-Small with SE and inverted residual blocks
    - Output: 10-dimensional embedding (paper requirement)
    """
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        # Start from torchvision mobilenet_v3_small (includes SE and residuals)
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = base.features   # convolution + 15 block modules
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Paper: after pooling, reduce to 1x1x10
        # torchvision v3_small ends with 576 channels -> map to 10
        self.conv1x1_to10 = nn.Conv2d(576, embedding_dim, kernel_size=1, bias=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)        # -> N x 576 x 7 x 7
        x = self.pool(x)            # -> N x 576 x 1 x 1
        x = self.conv1x1_to10(x)    # -> N x 10 x 1 x 1
        x = self.flatten(x)         # -> N x 10
        return x


class SiameseNet(nn.Module):
    """
    Siamese Network: two weight-sharing branches of MobileNetV3+ embedding
    """
    def __init__(self, embedding_dim=10, pretrained=True):
        super().__init__()
        self.embedding_net = MobileNetV3PlusEmbedding(embedding_dim, pretrained)

    def forward_once(self, x):
        return self.embedding_net(x)

    def forward(self, x1, x2):
        z1 = self.forward_once(x1)
        z2 = self.forward_once(x2)
        return z1, z2


def pairwise_l2(z1, z2):
    """
    Compute Euclidean distance between two embeddings
    """
    return torch.norm(z1 - z2, p=2, dim=1)
