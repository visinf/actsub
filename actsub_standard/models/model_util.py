from models.ood_resnet import resnet50
from models.ood_mobilenet import mobilenet_v2
import models.ood_densenet as dn
import torch

def build_model(id_data, model_name):
    if id_data == 'cifar':
        model = dn.DenseNet3(100, model_name, p = None, info = None)
        checkpoint = torch.load(r"checkpoints/densenet100_cifar" + str(model_name) + ".pth")
        model.load_state_dict(checkpoint)
        return model
    elif id_data == 'in1k':
        if model_name == 'resnet':
            model = resnet50(pretrained=True, num_classes = 1000)
        elif model_name == 'mobilenet':
            model = mobilenet_v2(pretrained=True, num_classes = 1000)
        return model

    else:
        raise Exception('Insert a valid ID data name. Select either cifar or in1k.')