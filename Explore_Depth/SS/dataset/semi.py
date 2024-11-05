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

        if mode in ['train_l', 'train_u', 'depth_l', 'depth_u']:  # 包括深度图模式
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode in ['train_l', 'depth_l'] and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('dataset/splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        # 对于深度图模式，只需要读取单一深度图
        if self.mode in ['depth_l', 'depth_u']:
            depth_img = Image.open(os.path.join(self.root, id)).convert('RGB')  # 读取RGB深度图
            depth_img = normalize(depth_img)  # 对深度图进行归一化
            return depth_img

        # 读取RGB图像、标签和深度图
        img_path, mask_path = id.split(' ')  # 分离出图像和标签路径
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')  # 读取RGB图像
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, mask_path))))  # 读取标签
        depth_img = Image.open(os.path.join(self.root.replace('img_512', 'depth_512'), img_path)).convert('RGB')  # 读取单通道深度图

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            depth_img = normalize(depth_img)  # 对深度图归一化
            return img, mask, depth_img, id

        # 统一对RGB图像、标签和深度图进行相同的处理
        img, mask, depth_img = resize(img, mask, depth_img, (0.5, 2.0))  # 统一缩放
        ignore_value = 255
        img, mask, depth_img = crop(img, mask, depth_img, self.size, ignore_value)  # 统一裁剪
        img, mask, depth_img = hflip(img, mask, depth_img, p=0.5)  # 统一水平翻转

        if self.mode == 'train_l':
            img, mask = normalize(img, mask)
            return img, mask  # 只返回图像和标签
        # 无标签图像的增强处理
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)
        img_s1 = color_transformation(img_s1)
        img_s2 = color_transformation(img_s2)

        # 归一化处理
        img_w = normalize(img_w)
        img_s1 = normalize(img_s1)
        img_s2 = normalize(img_s2)
        depth_img = normalize(depth_img)

        # 检查是否为tensor，若不是则转换
        to_tensor = transforms.ToTensor()
        if not isinstance(img_w, torch.Tensor):
            img_w = to_tensor(img_w)
        if not isinstance(img_s1, torch.Tensor):
            img_s1 = to_tensor(img_s1)
        if not isinstance(img_s2, torch.Tensor):
            img_s2 = to_tensor(img_s2)
        # if not isinstance(depth_img, torch.Tensor):
        #     depth_img = to_tensor(depth_img)

        return img_w, img_s1, img_s2 # 返回增强后的图像和归一化后的深度图

    def __len__(self):
        return len(self.ids)
