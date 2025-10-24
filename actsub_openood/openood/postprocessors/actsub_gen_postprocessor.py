from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from sklearn.cluster import KMeans


class ActSubGENPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ActSubGENPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tune_decisive = self.args.tune_decisive

        self.lmbd = self.args.lmbd
        self.gamma = self.args.gamma
        self.M = self.args.M
        self.is_proto = False
        self.c_idx = self.args.c_idx


        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.args_dict_decisive = self.config.postprocessor.postprocessor_sweep_decisive
        self.APS_mode = self.config.postprocessor.APS_mode

        self.hyperparam_search_done = False
        self.decisive_search_done = False


    def prep(self, net:nn.Module, t_embs:torch.Tensor):
        A = net.get_fc_layer().weight.data
        if self.c_idx == -1:
            self.c_idx = find_c_idx(A, t_embs)
        self.dec_transform, self.insig_transform = subspace_transforms(A, self.c_idx)
        self.insig_train_samples = t_embs@self.insig_transform.T
        self.linhead = net.get_fc_layer()

    
    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logit , emb = net.forward(data, return_feature=True)
        _, pred = torch.max(logit, dim=1)
        inf_args={  
            'dec_args':{
                'linhead': self.linhead,
                'dec_transform': self.dec_transform,
                'gamma' : self.gamma,
                'M': self.M

            },
            'insig_args':{
                'insig_transform': self.insig_transform,
                'insig_train_samples': self.insig_train_samples,
                'is_proto': self.is_proto

            }
        }

        if self.decisive_search_done:
            score = actsub_gen(emb, self.lmbd, inf_args)
        else:
            score = actsub_dec_gen(emb, **inf_args['dec_args'])
        return pred, score

    def set_prots(self, prots):
        self.insig_train_samples = prots

    def set_hyperparam(self, hyperparam: list):
        self.lmbd = hyperparam[0]

    def get_hyperparam(self):
        return self.lmbd

    def set_hyperparam_decisive(self, hyperparam: list):
        self.gamma = hyperparam[0]
        self.M = hyperparam[1]

    def get_hyperparam_decisive(self):
        return [self.gamma, self.M]


##Utility functions for ActSub##

def generalized_entropy(probs, gamma=0.1, M=100):
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                           dim=1)
        return -scores


def f_cossim(emb, embs, k = 10):
    n_emb = torch.nn.functional.normalize(emb, dim=1)
    n_embs = torch.nn.functional.normalize(embs, dim=1)
    cossim = torch.matmul(n_emb, n_embs.permute(1, 0))
    return torch.topk(cossim, k, dim = 1, largest=True)[0].mean(1)


#Calculate the parameter k from Equation 4.
def find_c_idx(A, embs):
    _, _, V_t = torch.linalg.svd(A)
    ratios = []
    for idx in range(A.shape[1]):
        dec_t, insig_t = split_transform(V_t, idx)
        dec_comp_t = embs@(dec_t.T)
        insig_comp_t = embs@(insig_t.T)
        ratios.append((torch.norm(dec_comp_t, dim=1).mean() - torch.norm(insig_comp_t, dim=1).mean()).abs())
    return torch.tensor(ratios).argmin(0)


#Extract the subspaces
def split_transform(M, idx):
    return M.T[:,:idx]@M[:idx,:], M.T[:,idx:]@M[idx:,:]  

def subspace_transforms(A, c_idx):
    _, _, V_t = torch.linalg.svd(A)
    return split_transform(V_t, c_idx)

#Score of the decisive component from Equation 9.
def actsub_dec_gen(x, linhead, dec_transform, gamma, M):
    x = x@dec_transform.T
    dec_logit = linhead(x)
    dec_sm = torch.nn.functional.softmax(dec_logit, dim=1)
    gen = generalized_entropy(dec_sm, gamma, M)
    return gen

#Score of the insignificant component from Equation 7.
def actsub_insig(x, insig_transform, insig_train_samples, is_proto):
    x = x@insig_transform.T
    if is_proto:
        insig_cossim = f_cossim(x, insig_train_samples, k=1)
    else:
        insig_cossim = f_cossim(x, insig_train_samples)
    log_cossim = -(1. - insig_cossim).log()
    return log_cossim


#ActSub score from Equation 10. Note: In contrast to Energy, Entropy is larger for OOD samples. Because of this, instead of multiplying by the insignificant score, we divide by it. 
def actsub_gen(x, lmbd, inf_args):
    dec_act = actsub_dec_gen(x, **inf_args['dec_args'])
    insig_score = actsub_insig(x, **inf_args['insig_args'])
    actsub_score = dec_act / (insig_score**lmbd)
    return actsub_score 