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

def generate_unsup_aug_sdc(conf_w, mask_w, data_s1, data_s2):
    b, _, im_h, im_w = data_s1.shape
    device = data_s1.device
    new_conf_w, new_mask_w, new_data_s = [], [], []
    for i in range(b):
        augmix_mask = generate_cutmix_mask([im_h, im_w]).to(device)
        new_conf_w.append((conf_w[i] * augmix_mask + conf_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        new_mask_w.append((mask_w[i] * augmix_mask + mask_w[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        if i % 2 == 0:
            new_data_s.append((data_s1[i] * augmix_mask + data_s2[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))
        else:
            new_data_s.append((data_s2[i] * augmix_mask + data_s1[(i + 1) % b] * (1 - augmix_mask)).unsqueeze(0))

    new_conf_w, new_mask_w, new_data_s = (torch.cat(new_conf_w), torch.cat(new_mask_w), torch.cat(new_data_s))

    return new_conf_w, new_mask_w, new_data_s


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



class Depth_slobal_power(nn.Module):
    def __init__(self, max_weight=5.0, epsilon=1e-8):
        super(Depth_slobal_power, self).__init__()
        self.max_weight = max_weight
        self.epsilon = epsilon

        sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                       [0, 0, 0],
                                       [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_kernel_x = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_kernel_y = nn.Parameter(sobel_kernel_y, requires_grad=False)

        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.laplacian_kernel = nn.Parameter(laplacian_kernel, requires_grad=False)

    def forward(self, depth):
        if depth.dim() == 4 and depth.size(1) == 3:

            depth = depth.mean(dim=1, keepdim=True)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        depth_normalized = depth


        depth_min = depth.view(depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        depth_max = depth.view(depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + self.epsilon)  # [B, 1, H, W]

        depth_padded = F.pad(depth_normalized, (1, 1, 1, 1), mode='reflect')  # 使用反射填充


        grad_x = F.conv2d(depth_padded, self.sobel_kernel_x.to(depth.device))
        grad_y = F.conv2d(depth_padded, self.sobel_kernel_y.to(depth.device))

        depth_grad_first_order = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        depth_laplacian = F.conv2d(depth_padded, self.laplacian_kernel.to(depth.device))
        depth_grad_second_order = torch.abs(depth_laplacian)

        scales = [1, 0.75, 0.5, 0.25]
        depth_grad_multiscale = depth_grad_first_order.clone()
        depth_grad_second_multiscale = depth_grad_second_order.clone()

        for scale in scales[1:]:
            depth_scaled = F.interpolate(depth_normalized, scale_factor=scale, mode='bilinear', align_corners=True)
            depth_padded_scaled = F.pad(depth_scaled, (1, 1, 1, 1), mode='reflect')


            grad_x_scaled = F.conv2d(depth_padded_scaled, self.sobel_kernel_x.to(depth.device))
            grad_y_scaled = F.conv2d(depth_padded_scaled, self.sobel_kernel_y.to(depth.device))
            grad_first_order_scaled = torch.sqrt(grad_x_scaled ** 2 + grad_y_scaled ** 2 + self.epsilon)
            grad_first_order_scaled = F.interpolate(grad_first_order_scaled, size=depth.size()[2:], mode='bilinear', align_corners=True)
            depth_grad_multiscale += grad_first_order_scaled


            laplacian_scaled = F.conv2d(depth_padded_scaled, self.laplacian_kernel.to(depth.device))
            grad_second_order_scaled = torch.abs(laplacian_scaled)
            grad_second_order_scaled = F.interpolate(grad_second_order_scaled, size=depth.size()[2:], mode='bilinear', align_corners=True)
            depth_grad_second_multiscale += grad_second_order_scaled


        depth_grad_multiscale = depth_grad_multiscale / len(scales)
        depth_grad_second_multiscale = depth_grad_second_multiscale / len(scales)


        combined_grad = depth_grad_multiscale + depth_grad_second_multiscale


        grad_min = combined_grad.view(depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        grad_max = combined_grad.view(depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        combined_grad_normalized = (combined_grad - grad_min) / (grad_max - grad_min + self.epsilon)


        weight_map = 1 + self.max_weight * torch.tanh(combined_grad_normalized)

        weight_map = weight_map.squeeze(1)

        return weight_map


def loss_calc(pred, label, weights, weight_map=None):
    """
    该函数返回用于语义分割的加权交叉熵损失。
    """
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()
    return criterion(pred, label, weights, weight_map)


class CrossEntropy2d_ignore(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None, weight_map=None):
        """
        Args:
            predict: (n, c, h, w) - 预测输出
            target: (n, h, w) - 真实标签
            weight: (c,) - 类别权重
            weight_map: (n, h, w) - 像素级权重图
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()


        target_mask = (target >= 0) & (target != self.ignore_label)
        valid_indices = target_mask.nonzero(as_tuple=True)

        if len(valid_indices[0]) == 0:
            return torch.zeros(1).cuda()


        predict = predict.permute(0, 2, 3, 1)  # (n, h, w, c)
        predict = predict[valid_indices]
        predict = predict.view(-1, c)  # (num_valid_pixels, c)
        target = target[valid_indices]  # (num_valid_pixels,)

        if weight_map is not None:
            weight_map = weight_map.to(predict.device)
            per_pixel_weight = weight_map[valid_indices]  # (num_valid_pixels,)
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



