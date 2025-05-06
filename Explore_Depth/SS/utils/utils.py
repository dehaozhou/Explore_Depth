import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from skimage import io
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output
from skimage.color import rgb2gray
from tqdm import tqdm
import tifffile
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from skimage import io
import os
import numpy as np
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms
import torch.nn.functional as F
from skimage.color import rgb2gray
from skimage.exposure import adjust_sigmoid, adjust_gamma
from skimage.filters import gaussian
from skimage.util import random_noise
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import pywt
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import pywt
import torch.nn as nn
from scipy import ndimage
from skimage.morphology import binary_dilation

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_cutmix_mask(img_size, ratio=2):
    cut_area = img_size[0] * img_size[1] / ratio
    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cut_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)
    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0

    return mask.long()

def generate_unsup_aug_sc(conf_w, mask_w, data_s):
    b, _, im_h, im_w = data_s.shape
    device = data_s.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s[i] * augmix_mask + data_s[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s

def generate_unsup_aug_ds(data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_data_s = []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[i] * (1 - augmix_mask)).unsqueeze(0))
    new_data_s = torch.cat(new_data_s)

    return new_data_s

def generate_unsup_aug_dc(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s



def generate_depth_sam(conf_w, mask_w, data_s1, data_s2, boundary_PL, object_PL):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s1, new_data_s2, new_boundary_PL, new_object_PL = [], [], [], [], [], []

    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        if i % 2 == 0:
            new_data_s1.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s1.append((data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_boundary_PL.append(
            (boundary_PL[i] * augmix_mask + boundary_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_object_PL.append((object_PL[i] * augmix_mask + object_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w = torch.cat(new_conf_w)
    new_mask_w = torch.cat(new_mask_w)
    new_data_s1 = torch.cat(new_data_s1)
    new_boundary_PL = torch.cat(new_boundary_PL)
    new_object_PL = torch.cat(new_object_PL)

    return new_conf_w, new_mask_w, new_data_s1, new_boundary_PL, new_object_PL








def generate_depth(conf_w, mask_w, data_s1, data_s2, boundary_PL, object_PL):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s1, new_data_s2, new_boundary_PL, new_object_PL = [], [], [], [], [], []

    for i in range(b):
        augmix_mask = generate_cutmix_mask_depth([im_h, im_w], boundary_PL[i].unsqueeze(0)).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

        if i % 2 == 0:
            new_data_s1.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s1.append((data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

        new_boundary_PL.append((boundary_PL[i] * augmix_mask + boundary_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_object_PL.append((object_PL[i] * augmix_mask + object_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

    new_conf_w = torch.cat(new_conf_w)
    new_mask_w = torch.cat(new_mask_w)
    new_data_s1 = torch.cat(new_data_s1)
    new_boundary_PL = torch.cat(new_boundary_PL)
    new_object_PL = torch.cat(new_object_PL)

    return new_conf_w, new_mask_w, new_data_s1, new_boundary_PL, new_object_PL


def generate_cutmix_mask_depth(img_size, boundary):
    mask = torch.ones(img_size, dtype=torch.float32)
    boundary = boundary.squeeze()
    boundary = torch.mean(boundary, dim=0)
    depth_min, depth_max = boundary.min(), boundary.max()
    depth_mid = (depth_min + depth_max) / 2
    mask[boundary > depth_mid] = 0
    mask[boundary <= depth_mid] = 1
    if np.random.rand() < 0.5:
        mask = 1 - mask
    if torch.sum(mask == 1) == 0 or torch.sum(mask == 0) == 0:
        mask = torch.ones(img_size, dtype=torch.float32)
        cut_area = img_size[0] * img_size[1] / 2
        w = np.random.randint(img_size[1] / 2 + 1, img_size[1])
        h = np.round(cut_area / w)
        x_start = np.random.randint(0, img_size[1] - w + 1)
        y_start = np.random.randint(0, img_size[0] - h + 1)
        x_end = int(x_start + w)
        y_end = int(y_start + h)
        mask[y_start:y_end, x_start:x_end] = 0 if torch.sum(mask == 1) == 0 else 1
    return mask.long()





def generate_sam(conf_w, mask_w, data_s1, data_s2, boundary_PL, object_PL):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s1, new_data_s2, new_boundary_PL, new_object_PL = [], [], [], [], [], []

    for i in range(b):
        augmix_mask = generate_cutmix_mask_freq([im_h, im_w],object_PL[i].unsqueeze(0)).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        if i % 2 == 0:
            new_data_s1.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s1.append((data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

        new_boundary_PL.append(
            (boundary_PL[i] * augmix_mask + boundary_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_object_PL.append((object_PL[i] * augmix_mask + object_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

    new_conf_w = torch.cat(new_conf_w)
    new_mask_w = torch.cat(new_mask_w)
    new_data_s1 = torch.cat(new_data_s1)
    new_boundary_PL = torch.cat(new_boundary_PL)
    new_object_PL = torch.cat(new_object_PL)

    return new_conf_w, new_mask_w, new_data_s1, new_boundary_PL, new_object_PL


def generate_cutmix_mask_freq(img_size, object_PL):
    object_PL_np = object_PL.cpu().numpy()
    object_PL_np = np.squeeze(object_PL_np, axis=0)
    coeffs = pywt.dwt2(object_PL_np, 'haar')
    LL, (LH, HL, HH) = coeffs
    LL_resized = nn.functional.interpolate(
        torch.from_numpy(LL).unsqueeze(0).unsqueeze(0).float(),
        size=(256, 256), mode='bilinear', align_corners=True
    ).squeeze(0).squeeze(0).numpy()
    H_resized = np.mean([
        nn.functional.interpolate(torch.from_numpy(LH).unsqueeze(0).unsqueeze(0).float(), size=(256, 256),
                                  mode='bilinear', align_corners=True).squeeze(0).squeeze(0).numpy(),
        nn.functional.interpolate(torch.from_numpy(HL).unsqueeze(0).unsqueeze(0).float(), size=(256, 256),
                                  mode='bilinear', align_corners=True).squeeze(0).squeeze(0).numpy(),
        nn.functional.interpolate(torch.from_numpy(HH).unsqueeze(0).unsqueeze(0).float(), size=(256, 256),
                                  mode='bilinear', align_corners=True).squeeze(0).squeeze(0).numpy()
    ], axis=0)
    mask_weight = np.zeros_like(LL_resized)
    mask_weight[LL_resized > np.percentile(LL_resized, 50)] = 0.2  # 较低频区域
    mask_weight[H_resized > np.percentile(H_resized, 50)] = 1.5  # 较高频区域
    mask = np.random.rand(*img_size) < mask_weight
    mask = mask.astype(np.float32)

    mask_high_freq = (mask_weight > 1).astype(np.float32)
    mask_high_freq = binary_dilation(mask_high_freq, selem=np.ones((5, 5))).astype(np.float32)  # 5x5结构元素的膨胀
    mask[mask_high_freq == 1] = 1

    mask = ndimage.gaussian_filter(mask, sigma=2)
    mask = (mask > 0.5).astype(np.float32)

    if np.random.rand() < 0.5:
        mask = 1 - mask

    return torch.from_numpy(mask).long()



def generate_areas_sam(conf_w, mask_w, data_s1, data_s2, boundary_PL, object_PL):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s1, new_data_s2, new_boundary_PL, new_object_PL = [], [], [], [], [], []

    for i in range(b):
        augmix_mask = generate_cutmix_mask_sam([im_h, im_w],object_PL[i].unsqueeze(0)).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        if i % 2 == 0:
            new_data_s1.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s1.append((data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_boundary_PL.append(
            (boundary_PL[i] * augmix_mask + boundary_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_object_PL.append((object_PL[i] * augmix_mask + object_PL[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
    new_conf_w = torch.cat(new_conf_w)
    new_mask_w = torch.cat(new_mask_w)
    new_data_s1 = torch.cat(new_data_s1)
    new_boundary_PL = torch.cat(new_boundary_PL)
    new_object_PL = torch.cat(new_object_PL)

    return new_conf_w, new_mask_w, new_data_s1, new_boundary_PL, new_object_PL


def generate_cutmix_mask_sam(img_size, object_PL):
    object_PL_np = object_PL.cpu().numpy()
    object_PL_np = np.squeeze(object_PL_np, axis=0)
    median_val = np.mean(object_PL_np)
    mask = (object_PL_np > median_val).astype(np.float32)
    if np.random.rand() < 0.5:
        mask = 1 - mask
    mask = torch.from_numpy(mask).long()

    return mask



def entropy_map(a, dim):
    em = - torch.sum(a * torch.log2(a + 1e-10), dim=dim)
    return em


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def color_map(dataset='GID-15'):
    cmap = np.zeros((256, 3), dtype='uint8')
    if dataset == 'GID-15':
        cmap[0] = np.array([200, 0, 0])
        cmap[1] = np.array([250, 0, 150])
        cmap[2] = np.array([200, 150, 150])
        cmap[3] = np.array([250, 150, 150])
        cmap[4] = np.array([0, 200, 0])
        cmap[5] = np.array([150, 250, 0])
        cmap[6] = np.array([150, 200, 150])
        cmap[7] = np.array([200, 0, 200])
        cmap[8] = np.array([150, 0, 250])
        cmap[9] = np.array([150, 150, 250])
        cmap[10] = np.array([250, 200, 0])
        cmap[11] = np.array([200, 200, 0])
        cmap[12] = np.array([0, 0, 200])
        cmap[13] = np.array([0, 150, 200])
        cmap[14] = np.array([0, 200, 250])

    elif dataset == 'iSAID':
        cmap[0] = np.array([0, 0, 63])
        cmap[1] = np.array([0, 63, 63])
        cmap[2] = np.array([0, 63, 0])
        cmap[3] = np.array([0, 63, 127])
        cmap[4] = np.array([0, 63, 191])
        cmap[5] = np.array([0, 63, 255])
        cmap[6] = np.array([0, 127, 63])
        cmap[7] = np.array([0, 127, 127])
        cmap[8] = np.array([0, 0, 127])
        cmap[9] = np.array([0, 0, 191])
        cmap[10] = np.array([0, 0, 255])
        cmap[11] = np.array([0, 191, 127])
        cmap[12] = np.array([0, 127, 191])
        cmap[13] = np.array([0, 127, 255])
        cmap[14] = np.array([0, 100, 155])

    elif dataset == 'MSL' or dataset == 'MER':
        cmap[0] = np.array([128, 0, 0])
        cmap[1] = np.array([0, 128, 0])
        cmap[2] = np.array([128, 128, 0])
        cmap[3] = np.array([0, 0, 128])
        cmap[4] = np.array([128, 0, 128])
        cmap[5] = np.array([0, 128, 128])
        cmap[6] = np.array([128, 128, 128])
        cmap[7] = np.array([64, 0, 0])
        cmap[8] = np.array([192, 0, 0])

    elif dataset == 'Vaihingen':
        cmap[0] = np.array([255, 255, 255])
        cmap[1] = np.array([0, 0, 255])
        cmap[2] = np.array([0, 255, 255])
        cmap[3] = np.array([0, 255, 0])
        cmap[4] = np.array([255, 255, 0])

    elif dataset == 'DFC22':
        cmap[0] = np.array([219, 95, 87])
        cmap[1] = np.array([219, 151, 87])
        cmap[2] = np.array([219, 208, 87])
        cmap[3] = np.array([173, 219, 87])
        cmap[4] = np.array([117, 219, 87])
        cmap[5] = np.array([123, 196, 123])
        cmap[6] = np.array([88, 177, 88])
        cmap[7] = np.array([0, 128, 0])
        cmap[8] = np.array([88, 176, 167])
        cmap[9] = np.array([153, 93, 19])
        cmap[10] = np.array([87, 155, 219])
        cmap[11] = np.array([0, 98, 255])


    return cmap


class Depth_MoE(nn.Module):
    """重构后的 Depth MoE，用于生成深度加权图，同时输出专家门控权重"""
    def __init__(self, win=3, embed_dim=16, num_heads=2):
        super().__init__()
        self.win = win
        self.embed_dim = embed_dim
        self.depth_proj = nn.Linear(1, embed_dim)
        self.semantic_proj = nn.Linear(1, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        def make_expert():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        self.expert_geo    = make_expert()
        self.expert_sem    = make_expert()
        self.expert_fusion = make_expert()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3),
            nn.Softmax(dim=-1)
        )
        self.proj_out = nn.Linear(embed_dim, 1)

        # init.xavier_uniform_(self.expert_geo[0].weight)
        # init.xavier_uniform_(self.expert_sem[0].weight)
        # init.xavier_uniform_(self.expert_fusion[0].weight)

    def forward(self, depth, semantic_pred):
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        sem_prob = semantic_pred.softmax(dim=1).max(dim=1, keepdim=True).values
        B, _, H, W = depth.shape
        depth = (depth - depth.mean(dim=[2,3], keepdim=True)) \
                / (depth.std(dim=[2,3], keepdim=True) + 1e-6)
        sem_prob = (sem_prob - sem_prob.mean(dim=[2,3], keepdim=True)) \
                   / (sem_prob.std(dim=[2,3], keepdim=True) + 1e-6)
        depth = depth.detach()
        depth_patch = einops.rearrange(
            depth, 'b 1 (h w1) (w w2) -> (b h w) (w1 w2) 1',
            w1=self.win, w2=self.win
        )
        sem_patch   = einops.rearrange(
            sem_prob, 'b 1 (h w1) (w w2) -> (b h w) (w1 w2) 1',
            w1=self.win, w2=self.win
        )
        depth_feat = self.depth_proj(depth_patch)    # [N, win*win, D]
        sem_feat   = self.semantic_proj(sem_patch)   # [N, win*win, D]
        x = depth_feat + sem_feat                    # [N,win*win,D]
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)             # [N,win*win,D]
        x = self.norm1(x + attn_out)
        geo_in    = x + depth_feat
        sem_in    = x + sem_feat
        fusion_in = x + depth_feat + sem_feat
        o_geo    = geo_in    + self.expert_geo(geo_in)
        o_sem    = sem_in    + self.expert_sem(sem_in)
        o_fusion = fusion_in + self.expert_fusion(fusion_in)
        expert_outputs = torch.stack([o_geo, o_sem, o_fusion], dim=-1)
        mean_stat = x.mean(dim=1)        # [N, D]
        max_stat  = x.max(dim=1).values  # [N, D]
        std_stat  = x.std(dim=1)         # [N, D]
        stats = torch.cat([mean_stat, max_stat, std_stat], dim=-1)  # [N,3D]
        gate_w = self.gate(stats)        # [N, 3]
        avg_gate = gate_w.mean(dim=0)    # [3]
        gate_w = gate_w.unsqueeze(1).unsqueeze(2)  # [N,1,1,3]
        moe = (expert_outputs * gate_w).sum(dim=-1)  # [N, win*win, D]
        moe = self.norm2(moe + x)                    # 残差 + 归一化
        y = torch.sigmoid(self.proj_out(moe))       # [N,win*win,1]
        weight_map = einops.rearrange(
            y, '(b h w) (w1 w2) 1 -> b 1 (h w1) (w w2)',
            b=B, h=H//self.win, w=W//self.win, w1=self.win, w2=self.win
        )
        return weight_map.squeeze(1), avg_gate





def loss_calc(pred, label, weights, weight_map=None):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()
    return criterion(pred, label, weights, weight_map)


class CrossEntropy2d_ignore(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None, weight_map=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) & (target != self.ignore_label)
        valid_indices = target_mask.nonzero(as_tuple=True)
        if len(valid_indices[0]) == 0:
            return torch.zeros(1).cuda()
        predict = predict.permute(0, 2, 3, 1)
        predict = predict[valid_indices]
        predict = predict.view(-1, c)
        target = target[valid_indices]
        if weight_map is not None:
            weight_map = weight_map.to(predict.device)
            per_pixel_weight = weight_map[valid_indices]
        else:
            per_pixel_weight = None
        loss = F.cross_entropy(predict, target, weight=weight, reduction='none')
        if per_pixel_weight is not None:
            loss = loss * per_pixel_weight
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Depth(nn.Module):
    def __init__(self, lambda_seg_depth=0.1, lambda_smooth=0.1, background_label=None):
        super(Depth, self).__init__()
        self.lambda_seg_depth = lambda_seg_depth
        self.lambda_smooth = lambda_smooth
        self.background_label = background_label
        self.criterion = CrossEntropy2d_ignore().cuda()

    def gradient_x(self, img):
        if img.dim() == 3:
            img = img.unsqueeze(1)
        gx = img[:, :, :, 1:] - img[:, :, :, :-1]
        gx = F.pad(gx, (0, 1, 0, 0), mode='replicate')
        return gx
    def gradient_y(self, img):
        if img.dim() == 3:
            img = img.unsqueeze(1)
        gy = img[:, :, 1:, :] - img[:, :, :-1, :]
        gy = F.pad(gy, (0, 0, 0, 1), mode='replicate')
        return gy

    def depth_smoothness_loss(self, pred_seg, true_depth):
        depth_grad_x = self.gradient_x(true_depth)
        depth_grad_y = self.gradient_y(true_depth)
        depth_smooth_weight = torch.exp(-torch.abs(depth_grad_x) - torch.abs(depth_grad_y))
        pred_prob = F.softmax(pred_seg, dim=1)
        seg_grad_x = self.gradient_x(pred_prob)
        seg_grad_y = self.gradient_y(pred_prob)
        smooth_loss = torch.mean(depth_smooth_weight * (torch.abs(seg_grad_x) + torch.abs(seg_grad_y)))
        return smooth_loss

    def seg_depth_alignment_loss(self, pred_seg, true_depth):
        depth_grad_x = self.gradient_x(true_depth)
        depth_grad_y = self.gradient_y(true_depth)
        depth_edge = torch.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2 + 1e-6)
        temperature = 0.1
        pred_prob = F.softmax(pred_seg / temperature, dim=1)
        seg_grad_x = self.gradient_x(pred_prob)
        seg_grad_y = self.gradient_y(pred_prob)
        seg_edge = torch.sqrt(seg_grad_x ** 2 + seg_grad_y ** 2 + 1e-6)
        alignment_loss = torch.mean(depth_edge * seg_edge)
        return alignment_loss

    def loss_calc(self, pred, label, weights, weight_map=None):
        label = Variable(label.long()).cuda()
        return self.criterion(pred, label, weights, weight_map)

    def forward(self, pred_seg, true_seg, true_depth, image, weights, weight_map):
        true_seg = true_seg.to(pred_seg.device)
        true_depth = true_depth.to(pred_seg.device)
        image = image.to(pred_seg.device)
        loss_weight = self.loss_calc(pred_seg, true_seg, weights, weight_map)
        total_loss = loss_weight 
        return total_loss


class SAM(nn.Module):
    def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

    def forward(self, pred, gt):
        gt_sam = gt.clone()
        if gt_sam.ndim == 4 and gt_sam.shape[3] == 3:
            gt_sam = gt_sam[..., 0]
        num_object = int(torch.max(gt_sam)) + 1
        num_object = min(num_object, self.max_object)
        total_object_loss = 0
        for object_index in range(1, num_object):
            mask = torch.where(gt_sam == object_index, 1, 0).unsqueeze(1).to('cuda')
            num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
            avg_pool = mask / (num_point + 1)
            object_feature = pred.mul(avg_pool)
            avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1, 1, gt_sam.shape[1], gt_sam.shape[2])
            avg_feature = avg_feature.mul(mask)
            object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
            total_object_loss = total_object_loss + object_loss
        return total_object_loss





