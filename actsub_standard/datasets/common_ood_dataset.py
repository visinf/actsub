import torch
import torch.utils.data
import os.path as osp
import os
from PIL import Image

class CommonOODDataset(torch.utils.data.Dataset):
    def __init__(self, dir, set_name, transform):
        if set_name == None:
            self.dir = osp.join(dir, r"imagenet/val")
        else:
            self.dir = osp.join(osp.join(dir, set_name), r"images")
        self.images = os.listdir(self.dir)
        self.transform = transform

    def __getitem__(self, index):
        im = Image.open(osp.join(self.dir, self.images[index])).convert('RGB')
        image = self.transform(im)
        return_dict =  {
            'image': image
        }
        return return_dict

    def __len__(self):
        return len(self.images)
    

class ImageNetSubDatasetVal(torch.utils.data.Dataset):
    def __init__(self, dir, class_list, transform):
        self.dir = dir
        self.class_list = class_list
        self.transform = transform
        if self.class_list == None:
            self.im_idxs =self._collect_im_ids(os.listdir(self.dir))
        else:
            self.im_idxs = self._collect_im_ids(self._read_class_ids(class_list))
        
    def _read_class_ids(self, class_list_f):
        with open(class_list_f) as file:
            class_list = [line.rstrip() for line in file]
        return class_list

    def _collect_im_ids(self, class_list):
        temp_ids = []
        for idx, class_ in enumerate(class_list):
            base_cdir = osp.join(self.dir, class_)
            temp_ids = temp_ids + [(osp.join(base_cdir, img_dir), idx) for img_dir in os.listdir(base_cdir)]
        return temp_ids

    def __getitem__(self, idx):
        img_dir, class_id = self.im_idxs[idx]
        im = Image.open(img_dir).convert('RGB')
        image = self.transform(im)
        return_dict =  {
            'image': image, 
            'c_id' : class_id
        }
        return return_dict

    def __len__(self):
        return len(self.im_idxs)
    


