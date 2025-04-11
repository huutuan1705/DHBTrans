import torch

def compute_inner_product(h):
    """
    Args:
        h (tensor): Hash code (N, B) B is length of vector hash code

    Returns:
        tensor: inner product
    """
    return torch.matmul(h, h.T)

def compute_similarity_matrix(labels):
    """
    Args:
        labels (tensor): tensor shape (N,), include all labels

    Returns:
        tensor: S matrix shape (N, N), S[i][j] = 1 if Ii and Ij same label, 0 otherwise
    """
    return (labels.unsqueeze(1) == labels.unsqueeze(0)).float()