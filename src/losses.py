import torch

def contrastive_loss(dist, labels, margin=0.60):
    """
    Contrastive Loss (Hadsell et al. 2006) used in the paper
    dist   : [N] tensor of L2 distances between pairs
    labels : [N] tensor (1 = same scribe, 0 = different)
    margin : margin hyperparameter (paper = 0.60)

    Loss = 0.5 * [ y * D^2 + (1-y) * max(0, margin-D)^2 ]
    """
    labels = labels.float()
    pos_loss = labels * torch.pow(dist, 2)
    neg_loss = (1 - labels) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    return 0.5 * (pos_loss + neg_loss).mean()


def triplet_loss(anchor, positive, negative, margin=0.6):
    """
    Triplet Loss (Schroff et al. 2015 - FaceNet)

    anchor, positive, negative: [N, D] tensors of embeddings
    margin: margin hyperparameter (default 1.0)

    Loss = mean( max(0, D(a,p)^2 - D(a,n)^2 + margin) )
    """
    # Compute pairwise L2 distances
    d_pos = torch.norm(anchor - positive, p=2, dim=1)
    d_neg = torch.norm(anchor - negative, p=2, dim=1)

    # Triplet margin loss
    losses = torch.clamp(d_pos - d_neg + margin, min=0.0)
    return losses.mean()