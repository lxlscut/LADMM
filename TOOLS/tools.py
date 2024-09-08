import torch
import torch.nn.functional as F


def cos_sim(data):
    # Flatten the data if necessary
    data = data.view(data.size(0), -1)
    # Normalize the data along the feature dimension
    data = F.normalize(data, p=2, dim=1)
    # Compute the cosine similarity matrix
    return torch.mm(data, data.transpose(0, 1))

