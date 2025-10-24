import argparse
import torch
import numpy as np
import os
from methods import *
from models.model_util import build_model
from datasets.dataset_util import build_dataset
from ood_utils import evaluate, load_config_yml, collect_embs


def run_inf(config):
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
    pruning_p = config['pruning_p']
    lmbd = config['lmbd']
    c_idx = config['c_idx']


    #Load model.
    b_model = build_model(id_data, model_name).to(device).eval()
    fc_weights = b_model.return_fc_weights()

    #Build ID and OOD datasets.
    if id_data == 'in1k':
        id_dataset = build_dataset(data_root, id_data, id_data)
        training_samples = torch.tensor(np.load(sample_dir)).to(device)
        ood_dataset_name = ['iNaturalist', 'SUN', 'Places', 'Textures']
    else:
        train_dataset, id_dataset = build_dataset(data_root, id_data, id_data + str(model_name))
        training_samples = collect_embs(train_dataset, b_model, batch_size, device)
        ood_dataset_name = ['SVHN', 'iSUN', 'Places365', 'Textures']




    ood_dataset = []
    for name in ood_dataset_name:
        ood_dataset.append({
            "name": name,
            "dataloader" : build_dataset(data_root, id_data, name)
        })

    
    

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

    evaluate(id_dataset, ood_dataset, batch_size, **inf_args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    args = parser.parse_args()
    config = load_config_yml(args.config)
    run_inf(config)
    