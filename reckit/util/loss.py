import torch
import torch.nn.functional as F

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

def InfoNCE(view1, view2, temperature):
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.sum(cl_loss)
