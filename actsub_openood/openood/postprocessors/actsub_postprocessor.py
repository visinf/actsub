from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from openood.networks.scale_net import scale
from openood.networks.ash_net import ash_s


class ActSubPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(ActSubPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tune_decisive = self.args.tune_decisive

        self.func = self.args.func
        self.lmbd = self.args.lmbd
        self.pp = self.args.pp
        self.is_proto = self.args.is_proto
        self.c_idx = self.args.c_idx


        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.args_dict_decisive = self.config.postprocessor.postprocessor_sweep_decisive
        self.APS_mode = self.config.postprocessor.APS_mode


        self.decisive_search_done = False
        self.hyperparam_search_done = False

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
                'func': eval(self.func),
                'percentile' : self.pp

            },
            'insig_args':{
                'insig_transform': self.insig_transform,
                'insig_train_samples': self.insig_train_samples,
                'is_proto': self.is_proto

            }
        }

        if self.decisive_search_done:
            score = actsub(emb, self.lmbd, inf_args)
        else:
            score = actsub_dec(emb, **inf_args['dec_args'])
        return pred, score

    def set_prots(self, prots):
        self.insig_train_samples = prots

    def set_hyperparam(self, hyperparam: list):
        self.lmbd = hyperparam[0]

    def get_hyperparam(self):
        return self.lmbd
    
    def set_hyperparam_decisive(self, hyperparam: list):
        self.pp = hyperparam[0]

    def get_hyperparam_decisive(self):
        return self.pp


##Utility functions for ActSub##

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
def actsub_dec(x, linhead, dec_transform, func, percentile):
    x = x@dec_transform.T
    dec_act = func(x.unsqueeze(2).unsqueeze(2), percentile=percentile).squeeze()
    dec_logit = linhead(dec_act)
    dec_energy = torch.logsumexp(dec_logit, dim=1)
    return dec_energy

#Score of the insignificant component from Equation 7.
def actsub_insig(x, insig_transform, insig_train_samples, is_proto):
    x = x@insig_transform.T
    if is_proto:
        insig_cossim = f_cossim(x, insig_train_samples, k=1)
    else:
        insig_cossim = f_cossim(x, insig_train_samples)
    log_cossim = -(1. - insig_cossim).log()
    return log_cossim


#ActSub score from Equation 10.
def actsub(x, lmbd, inf_args):
    dec_act = actsub_dec(x, **inf_args['dec_args'])
    insig_score = actsub_insig(x, **inf_args['insig_args'])
    actsub_score = dec_act * insig_score**lmbd
    return actsub_score 