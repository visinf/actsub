import sys
sys.path.insert(0, "actsub_openood")
sys.path.insert(0, "actsub_standard")

import argparse
import torch
import numpy as np
import os
from torch.hub import load_state_dict_from_url
from torchvision.models import ViT_B_16_Weights, ResNet50_Weights
from tqdm import tqdm

from actsub_standard.models.model_util import build_model
from actsub_standard.datasets.dataset_util import build_dataset
from actsub_standard.ood_utils import load_config_yml, collect_embs, dataloader_test

from actsub_openood.openood.networks import ResNet50, ViT_B_16

def collect_embs_openood(train_dataloader, b_model, device):
    for i, imgs in enumerate(tqdm(train_dataloader)):
        with torch.no_grad():
            _, emb = b_model.forward(imgs["image"].to(device), return_feature=True)
        if i == 0:
            cat_emb = emb
        else:
            cat_emb = torch.cat([cat_emb, emb], dim=0)

    idx = torch.randperm(cat_emb.size(0), device = cat_emb.device)[:cat_emb.shape[0]//10]
    return cat_emb[idx]

def extract(config):
    if config['device'] == -99:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(config['device']))

    batch_size = config['batch_size']
    data_root = os.path.expandvars(config["data_root"])
    experiment = config['experiment']
    model = config['model']

    if experiment == "standard":
        train_dataset = build_dataset(data_root, "in1k-t", r"actsub_standard/checkpoints/base_list_c.txt")
        b_model = build_model("in1k", model).to(device).eval()
        train_embs = collect_embs(train_dataset, b_model, batch_size, device)
        path = r"actsub_standard/checkpoints/" + "in1k_" + model + "_train_samples.npy"
        with open(path, 'wb') as f:
            np.save(f, train_embs.detach().cpu().numpy())


    elif experiment == "openood":
        train_dataset = build_dataset(data_root, "in1k-t", r"actsub_standard/checkpoints/base_list_c.txt")
        train_dataloader = dataloader_test(train_dataset, batch_size)
        if model == "resnet":
            b_model = ResNet50()
            weights = ResNet50_Weights.IMAGENET1K_V1
        elif model == "vit":
            b_model = ViT_B_16()
            weights = ViT_B_16_Weights.IMAGENET1K_V1
        b_model.load_state_dict(load_state_dict_from_url(weights.url))
        b_model.to(device)
        b_model.eval()
        resnet_embs = collect_embs_openood(train_dataloader, b_model, device)
        path_resnet = r"actsub_openood/checkpoints/" + "in1k_" + model + "_train_samples.npy"
        with open(path_resnet, 'wb') as f:
            np.save(f, resnet_embs.detach().cpu().numpy())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    args = parser.parse_args()
    config = load_config_yml(args.config)
    extract(config)