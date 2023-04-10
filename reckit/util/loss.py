import torch

def l2_loss(*weights):
    """Compute  the L2 norm of tensors without the `sqrt`
    Args:
        weights: Variable length weight list.
    """
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2))
    return 0.5 * loss
