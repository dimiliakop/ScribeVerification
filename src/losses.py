import torch

def contrastive_loss(dist, labels, margin=0.60):
    labels = labels.float()
    pos_loss = labels * torch.pow(dist, 2)
    neg_loss = (1 - labels) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    return 0.5 * (pos_loss + neg_loss).mean()


def triplet_loss(anchor, positive, negative, margin=0.6):
    # Compute pairwise L2 distances
    d_pos = torch.norm(anchor - positive, p=2, dim=1)
    d_neg = torch.norm(anchor - negative, p=2, dim=1)

    # Triplet margin loss
    losses = torch.clamp(d_pos - d_neg + margin, min=0.0)
    return losses.mean()