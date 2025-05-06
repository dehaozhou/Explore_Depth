from dataset.transform import crop, hflip, normalize, resize, color_transformation
import numpy as np
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
import torch

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('./dataset/splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        if self.mode == 'val':
            img_path, label_path = id.split(' ')
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, label_path))))
        else:  
            img_path, label_path, depth_path, sam_path = id.split(' ')
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, label_path))))
            depth_img = Image.open(os.path.join(self.root, depth_path)).convert('L')
            sam_img = Image.open(os.path.join(self.root, sam_path)).convert('L')

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id
        img, mask, depth_img, sam_img  = resize(img, mask, depth_img,sam_img, (0.5, 2.0))  # 统一缩放
        ignore_value = 255
        img, mask, depth_img,sam_img  = crop(img, mask, depth_img,sam_img , self.size, ignore_value)  # 统一裁剪
        img, mask, depth_img,sam_img  = hflip(img, mask, depth_img,sam_img, p=0.5)  # 统一水平翻转
        if self.mode == 'train_l':
            img, mask = normalize(img, mask)
            sam_img = torch.from_numpy(np.array(sam_img)).long()
            depth_img = torch.from_numpy(np.array(depth_img)).float()
            return img, mask, depth_img, sam_img
        
        # 无标签图像的增强处理
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        img_s1 = color_transformation(img_s1)
        img_s2 = color_transformation(img_s2)
        # 归一化处理
        img_w = normalize(img_w)
        img_s1 = normalize(img_s1)
        img_s2 = normalize(img_s2)
        sam_img = torch.from_numpy(np.array(sam_img)).long()
        depth_img = torch.from_numpy(np.array(depth_img)).float()

        return img_w, img_s1, img_s2,depth_img,sam_img

    def __len__(self):
        return len(self.ids)
