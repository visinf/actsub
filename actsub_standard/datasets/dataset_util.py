from datasets.cifar import CIFAR10, CIFAR100
from datasets.common_ood_dataset import CommonOODDataset, ImageNetSubDatasetVal
from datasets.svhn_dataset import SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os.path as osp

cifar_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
in1k_transform = transforms.Compose([

            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def build_dataset(dir, id_data, data_name):

    
    if id_data == 'cifar':
        if data_name == 'cifar10':
            train_dataset = CIFAR10(root=dir, train=True, transform=cifar_transform, download=True)
            id_dataset = CIFAR10(root=dir, train=False, transform=cifar_transform, download=True)
            return train_dataset, id_dataset
        elif data_name == 'cifar100':
            train_dataset = CIFAR100(root=dir, train=True, transform=cifar_transform, download=True)
            id_dataset = CIFAR100(root=dir, train=False, transform=cifar_transform, download=True)
            return train_dataset, id_dataset
        elif data_name == 'SVHN':
            dataset = SVHN(dir, split = 'test', download=True, transform=cifar_transform)
            return dataset
        else:
            dataset = CommonOODDataset(dir, data_name, transform=cifar_transform)
            return dataset

    elif id_data == 'in1k':
        if data_name == 'in1k':
            dataset = CommonOODDataset(dir, None, transform=in1k_transform)
        else:
            dataset = CommonOODDataset(dir, data_name, transform=in1k_transform)
        return dataset
    elif id_data == "in1k-r":
        dataset = ImageNetSubDatasetVal(osp.join(dir, r"imagenet-r"), data_name, transform=in1k_transform)
        return dataset
    elif id_data == "in1k-t":
        dataset = ImageNetSubDatasetVal(dir, data_name, transform=in1k_transform)
        return dataset
    else:
        raise Exception('Insert a valid ID data name. Select either cifar or in1k.')

def dataloader_test(dataset, batch_size, num_worker = 8):
    dataloder = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_worker,
    drop_last=False
    )
    return dataloder
