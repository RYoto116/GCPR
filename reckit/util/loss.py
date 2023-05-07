import torch

def l2_loss(*weights):
    loss = 0.0
    for w in weights:
        loss += torch.sum(torch.pow(w, 2))
    return 0.5 * loss

def cpr_loss(pos_scores, neg_scores, batch_size, sample_rate):
    cpr_obj = pos_scores - neg_scores
    if sample_rate == 1:
        cpr_obj_neg = -cpr_obj
    else:
        cpr_obj_neg, _ = torch.topk(-cpr_obj, k=batch_size, sorted=False)
    loss = torch.sum(torch.nn.functional.softplus(cpr_obj_neg))
    return loss
