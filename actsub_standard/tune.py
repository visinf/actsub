import argparse
import torch
import numpy as np
from methods import *
import os
from models.model_util import build_model
from datasets.dataset_util import build_dataset
from ood_utils import tune, load_config_yml, collect_embs


def run_tune(config):
    if config['device'] == -99:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(config['device']))

    #Load config variables.
    
    batch_size = config['batch_size']
    id_data = config['id_data']
    model_name = config['model_name']
    data_root = os.path.expandvars(config["data_root"])
    sample_dir = config['sample_dir']
    ood_func = config['ood_func']

    #Method parameters.
    pruning_p = config['pruning_p_tune']
    lmbd = config['lmbd_tune']
    c_idx = config['c_idx']


    #Load model.
    b_model = build_model(id_data, model_name).to(device).eval()
    fc_weights = b_model.return_fc_weights()

    #Build ID and OOD datasets.
    if id_data == 'in1k':
        id_dataset = build_dataset(data_root, id_data, id_data)
        id_dataset_r = build_dataset(data_root, "in1k-r", r"checkpoints/inr_c.txt")
        training_samples = torch.tensor(np.load(sample_dir)).to(device)
        ood_dataset = build_dataset(data_root, id_data, "NINCO")
        id_datasets = [id_dataset, id_dataset_r]
    else:
        train_dataset, id_dataset = build_dataset(data_root, id_data, id_data + str(model_name))
        training_samples = collect_embs(train_dataset, b_model, batch_size, device)
        ood_dataset = build_dataset(data_root, id_data, "MNIST")
        id_datasets = [id_dataset]

        
    #Due to the lack of a non-semantic variation dataset such as ImageNet-R for CIFAR, we use the same parameters reported by Djurisic et al.,"Extremely Simple Activation Shaping for Out-of-Distribution Detection, In ICLR 2023".
    #For an explanation of why a non-semantic variation dataset is required for tuning the decisive component, please refer to "Sensitivity of the decisive component." in the Supplementary Material. 
    if id_data == "cifar":
        if model_name == 10:
            pruning_p = [95]
        elif model_name == 100:
            pruning_p = [90]



    #Set the hyperparameter k from Equation 10.
    if c_idx == -1:
        c_idx = find_c_idx(fc_weights, training_samples)

    #Subspace transformation matrices for ActSub. 
    decide_trans, insig_trans = subspace_transforms(fc_weights, c_idx)
    insig_train_samples = training_samples@insig_trans.T

    inf_args = {
    'model': b_model,
    'ood_func': eval(ood_func),
    'device' : device,
    'dec_trans': decide_trans,
    'insig_trans': insig_trans,
    'insig_train_samples': insig_train_samples,
    'percentile': pruning_p,
    'lmbd': lmbd
    }

    tune(id_datasets, ood_dataset, batch_size, **inf_args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    args = parser.parse_args()
    config = load_config_yml(args.config)
    run_tune(config)
    