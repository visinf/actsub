import torch
import numpy as np


def f_cossim(emb, embs, k = 10):
    n_emb = torch.nn.functional.normalize(emb, dim=1)
    n_embs = torch.nn.functional.normalize(embs, dim=1)
    cossim = torch.matmul(n_emb, n_embs.permute(1, 0))
    return torch.topk(cossim, k, dim = 1, largest=True)[0].mean(1)


#find_c_idx: Find the parameter k specified in Equations 4, 6.
def find_c_idx(A, embs):
    _, _, V_t = torch.linalg.svd(A)
    ratios = []
    for idx in range(A.shape[1]):
        dec_t, insig_t = split_transform(V_t, idx)
        dec_comp_t = embs@(dec_t.T)
        insig_comp_t = embs@(insig_t.T)
        ratios.append((torch.norm(dec_comp_t, dim=1).mean() - torch.norm(insig_comp_t, dim=1).mean()).abs())
    return torch.tensor(ratios).argmin(0)


#Functions for extracting the decisive and the insignificant supspaces with the given index. 
def split_transform(M, idx):
    return M.T[:,:idx]@M[:idx,:], M.T[:,idx:]@M[idx:,:]  

def subspace_transforms(A, c_idx):
    _, _, V_t = torch.linalg.svd(A)
    return split_transform(V_t, c_idx)



#Score function of the decisive component from Equation 9.
def actsub_dec(x, **kwargs):
    x = x@kwargs['dec_trans'].T
    dec_act = scale_f(x.unsqueeze(2).unsqueeze(2), percentile=kwargs['percentile']).squeeze()
    dec_logit = kwargs['model'].forward_ood_linear(dec_act.squeeze())
    dec_energy = torch.logsumexp(dec_logit, dim=1)
    return dec_energy


#Score function of the insignificant component from Equation 7.
def actsub_insig(x, **kwargs):
    x = x@kwargs['insig_trans'].T
    insig_cossim = f_cossim(x, kwargs['insig_train_samples'])
    log_cossim = -(1. - insig_cossim).log()
    return log_cossim


#ActSub score function from Equation 10. 
def actsub(x, **kwargs):
    insig_score = actsub_insig(x, **kwargs)
    dec_act = actsub_dec(x, **kwargs)
    actsub_score = dec_act * insig_score**kwargs['lmbd']
    return actsub_score    


#ASH-S function without the subspace transformation.
def ash_s(x, **kwargs):
    dec_act = ash_s_f(x.unsqueeze(2).unsqueeze(2), percentile=kwargs['percentile']).squeeze()
    dec_logit = kwargs['model'].forward_ood_linear(dec_act.squeeze())
    dec_energy = torch.logsumexp(dec_logit, dim=1)
    return dec_energy

#SCALE function without the subspace transformation. 
def scale(x, **kwargs):
    dec_act = scale_f(x.unsqueeze(2).unsqueeze(2), percentile=kwargs['percentile']).squeeze()
    dec_logit = kwargs['model'].forward_ood_linear(dec_act.squeeze())
    dec_energy = torch.logsumexp(dec_logit, dim=1)
    return dec_energy


#Function that applies ASH-S to the extracted activations (https://github.com/andrijazz/ash). 
def ash_s_f(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=[1, 2, 3])
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x

#Function that applies SCALE to the extracted activations (https://github.com/kai422/SCALE). 
def scale_f(x, percentile=65):
    
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=[1, 2, 3])
    scale = s1 / s2
    
    return input * torch.exp(scale[:, None, None, None])

