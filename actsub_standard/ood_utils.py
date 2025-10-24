import torch
import numpy as np
from tqdm import tqdm
from datasets.dataset_util import dataloader_test
from datasets.common_ood_dataset import CommonOODDataset
from methods import *
import yaml

#Function for calculating the FPR and AUROC metrics. We follow the standard implementation (https://github.com/andrijazz/ash). 
def get_curve(known, novel, method=None):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()


    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    auc = -np.trapz(1.-fpr, tpr)
    return auc*100,  fpr_at_tpr95*100

def load_config_yml(config_file):
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            config = config['config']
            return config
    except:
        print('Config file {} is missing'.format(config_file))
        exit(1)

def collect_score(dataloader, **kwargs):
    for i, imgs in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            x = kwargs['model'].forward_ood_feats(imgs["image"].to(kwargs['device']))
            score = kwargs['ood_func'](x, **kwargs)
        score = score.detach().cpu()
        if i == 0:
            cat_score = score
        else:
            cat_score = torch.cat([cat_score, score], dim=0) 

    return cat_score.numpy()

def evaluate(id_dataset, ood_datasets, batch_size, **kwargs):
    id_dataloader = dataloader_test(id_dataset, batch_size)
    id_score = collect_score(id_dataloader, **kwargs)
    AUROCS = []
    FPRS = []
    for dataset in ood_datasets:
        ood_dataloader_t = dataloader_test(dataset["dataloader"], batch_size)
        ood_score_t = collect_score(ood_dataloader_t, **kwargs)
        AUROC, FPR = get_curve(id_score, ood_score_t)
        AUROCS.append(AUROC)
        FPRS.append(FPR)
        print(dataset["name"] + ": \n" + " AUROC:" + str(np.round(AUROC, 2)) + " FPR:" + str(np.round(FPR, 2)) )
    auroc_avg = np.array(AUROCS).mean()
    fpr_avg = np.array(FPRS).mean()
    print("Average:\n" + " AUROC:" + str(np.round(auroc_avg, 2)) + " FPR:" + str(np.round(fpr_avg, 2)) )

def collect_embs(train_dataset, model, batch_size, device, is_save=False):
    train_dataloader = dataloader_test(train_dataset, batch_size)
    for i, imgs in enumerate(tqdm(train_dataloader)):
        with torch.no_grad():
            emb = model.forward_ood_feats(imgs["image"].to(device))
        if i == 0:
            cat_emb = emb
        else:
            cat_emb = torch.cat([cat_emb, emb], dim=0)

    idx = torch.randperm(cat_emb.size(0), device = cat_emb.device)[:cat_emb.shape[0]//10]
    return cat_emb[idx]


#Tuning function for the decisive component.

def tune_decisive(id_datasets, ood_dataset, batch_size, pruning_p, **kwargs):
    kwargs["percentile"] = pruning_p
    ood_dataloader = dataloader_test(ood_dataset, batch_size)
    ood_score = collect_score(ood_dataloader, **kwargs)
    t_scores = []
    for id_dataset in id_datasets:
        id_dataloader_t = dataloader_test(id_dataset, batch_size)
        id_score_t = collect_score(id_dataloader_t, **kwargs)
        #Standard evaluation script assumes detecting ID's in get_curve. Only for tuning, we here assume detecting OOD's as it more closely resembles the task definition.
        #For fair comparison, for evaluating, we follow the existing work. 
        AUROC, FPR = get_curve(-ood_score, -id_score_t)
        t_scores.append(AUROC-FPR)
    return np.array(t_scores).mean()

#Tuning function for the insignificant component.

def tune_insig(id_datasets, ood_dataset, batch_size, lmbd_t, **kwargs):
    kwargs["lmbd"] = lmbd_t
    id_dataloader =  dataloader_test(id_datasets[0], batch_size)
    ood_dataloader = dataloader_test(ood_dataset, batch_size)
    id_score = collect_score(id_dataloader, **kwargs)
    ood_score = collect_score(ood_dataloader, **kwargs)
    AUROC, FPR = get_curve(-ood_score, -id_score)
    return AUROC


def tune(id_dataset, ood_dataset, batch_size, **kwargs):
    tune_args = {
        'model': kwargs["model"],
        'device': kwargs["device"],
        'ood_func': actsub_dec,
        'dec_trans': kwargs["dec_trans"],
        'insig_trans': kwargs["insig_trans"],
        'insig_train_samples': kwargs["insig_train_samples"],
    }
    decisive_p_range = kwargs["percentile"]
    insig_p_range =  kwargs["lmbd"]

    decisive_scores = []
    for p_percentage in decisive_p_range:
        found_score = tune_decisive(id_dataset, ood_dataset, batch_size, p_percentage, **tune_args)
        decisive_scores.append(found_score)
        print("Pruning percentage:" + str(p_percentage) + " AUROC-FPR:" + str(found_score))
    decisive_index = np.array(decisive_scores).argmax()
    found_pruning_p = decisive_p_range[decisive_index]
    tune_args["percentile"] = found_pruning_p
    tune_args["ood_func"] = actsub

    lmbd_scores = []
    for lmbd_t in insig_p_range:
        found_l_score = tune_insig(id_dataset, ood_dataset, batch_size, lmbd_t, **tune_args)
        lmbd_scores.append(found_l_score)
        print("Lambda:" + str(lmbd_t) + " AUROC:" + str(found_l_score))
    found_lmbd = insig_p_range[np.array(lmbd_scores).argmax()]
    print("Best pruning percentage: ", str(found_pruning_p))
    print("Best Lambda: ", str(found_lmbd))













