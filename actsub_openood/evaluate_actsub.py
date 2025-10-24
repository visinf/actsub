import os
import torch
import argparse
from openood.evaluation_api import Evaluator
from openood.networks import ResNet50, ViT_B_16
from torchvision import transforms as trn
from openood.postprocessors import ActSubPostprocessor, ActSubGENPostprocessor
import numpy as np
from openood.utils.config import Config
from torch.hub import load_state_dict_from_url
from torchvision.models import ViT_B_16_Weights, ResNet50_Weights
from sklearn.cluster import KMeans
import yaml


def load_config_yml(config_file):
    try:
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            config = config['config']
            return config
    except:
        print('Config file {} is missing'.format(config_file))
        exit(1)


def evaluate(config_common):
    model = config_common['model']
    data_dir = os.path.expandvars(config_common['data_dir'])
    is_gen = config_common['is_gen']
    is_proto = config_common['is_proto']


    if is_gen:
        config = Config("configs/postprocessors/actsub_gen.yml")
        postprocessor = ActSubGENPostprocessor(config)
    else:
        config = Config("configs/postprocessors/actsub.yml")
        postprocessor = ActSubPostprocessor(config)

    if model == "resnet":
        b_model = ResNet50()
        weights = ResNet50_Weights.IMAGENET1K_V1
        training_samples = torch.tensor(np.load(r"checkpoints/in1k_resnet_train_samples.npy")).cuda()

    elif model == "vit":
        b_model = ViT_B_16()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        training_samples = torch.tensor(np.load(r"checkpoints/in1k_vit_train_samples.npy")).cuda()

    b_model.load_state_dict(load_state_dict_from_url(weights.url))
    b_model.cuda()
    b_model.eval()
    preprocessor = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
    postprocessor.prep(b_model, training_samples.cuda())

    #Define the evaluator.
    evaluator = Evaluator(
    b_model,
    id_name='imagenet',
    data_root=data_dir,
    config_root="configs/",
    preprocessor=preprocessor,
    postprocessor_name="actsub", #Set the name to "actsub" for ActSub, ignore warning. 
    postprocessor=postprocessor, 
    batch_size=200,
    shuffle=False,
    num_workers=8)


    if is_proto:
        
        ncentroids = training_samples.shape[0]//100
        kmeans = KMeans(n_clusters=ncentroids, random_state=42, max_iter=20, n_init='auto')
        kmeans.fit(torch.nn.functional.normalize(postprocessor.insig_train_samples, dim=1).cpu())
        prots = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).cuda()    

        evaluator.postprocessor.is_proto=True
        evaluator.postprocessor.set_prots(prots)

    metrics = evaluator.eval_ood(fsood=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config YML")
    args = parser.parse_args()
    config_common = load_config_yml(args.config)
    evaluate(config_common)