# from dataset.semi import SemiDataset
# from model.semseg.deeplabv2 import DeepLabV2
# from utils.dataset_name import *
# from utils.utils import count_params, meanIOU, color_map

# import argparse
# from copy import deepcopy
# import numpy as np
# import os
# from PIL import Image
# import torch
# from torch.nn import CrossEntropyLoss, DataParallel
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# NUM_CLASSES = {'DFC22': 12, 'iSAID': 15, 'MER': 9, 'MSL': 9, 'Vaihingen': 5, 'GID-15': 15}
# DATASET = 'DFC22'     # ['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15']
# WEIGHTS = '/data5/WSCL_depth/exp/DFC22/1-8_20_1/SDC/deeplabv2_resnet101_34.84.pth'

# DFC22_DATASET_PATH = '/data5/WSCL_depth/dataset/splits/DFC22/'
# iSAID_DATASET_PATH = 'Your local path'
# MER_DATASET_PATH = 'Your local path'
# MSL_DATASET_PATH = 'Your local path'
# Vaihingen_DATASET_PATH = 'Your local path'
# GID15_DATASET_PATH = 'Your local path'

# def parse_args():
#     parser = argparse.ArgumentParser(description='WSCL Framework')

#     # basic settings
#     parser.add_argument('--data-root', type=str, default=None)
#     parser.add_argument('--dataset', type=str, choices=['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15'],
#                         default=DATASET)
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
#     parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
#                         default='deeplabv2')
#     parser.add_argument('--save-path', type=str, default='/data5/open_code/test/ssl/WSCL/0.125/yuanshi       /' + WEIGHTS.split('/')[-1].replace('.pth', ''))

#     args = parser.parse_args()
#     return args

# def create_path(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def DFC22():
#     class_names = ['Urban fabric', 'Industrial', 'Mine', 'Artificial', 'Arable', 'Permanent crops',
#                    'Pastures', 'Forests', 'Herbaceous', 'Open spaces', 'Wetlands', 'Water']

#     palette = [[219, 95, 87], [219, 151, 87], [219, 208, 87], [173, 219, 87], [117, 219, 87], [123, 196, 123],
#                [88, 177, 88], [0, 128, 0], [88, 176, 167], [153, 93, 19], [87, 155, 219], [0, 98, 255]]

#     return class_names, palette

# def apply_palette(pred, palette):
#     color_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
#     for class_idx, color in enumerate(palette):
#         color_img[pred == class_idx] = color
#     return color_img

# def main(args):
#     create_path(args.save_path)

#     model = DeepLabV2(args.backbone, NUM_CLASSES[args.dataset])
#     model.load_state_dict(torch.load(WEIGHTS))
#     model = DataParallel(model).cuda()

#     valset = SemiDataset(args.dataset, args.data_root, 'val', None)
#     valloader = DataLoader(valset, batch_size=8,
#                            shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
#     eval(model, valloader, valset, args)

# def eval(model, valloader, valset, args):
#     model.eval()
#     tbar = tqdm(valloader)
#     _, palette = DFC22()  # Load palette

#     data_list = []

#     with torch.no_grad():
#         for idx, (img, mask, depth_img, file_name) in enumerate(tbar):
#             img = img.cuda()
#             pred = model(img)
#             pred = torch.argmax(pred, dim=1).cpu().numpy()

#             # Save each prediction as a color image with the original file name (only the basename)
#             for i, p in enumerate(pred):
#                 color_pred = apply_palette(p, palette)
#                 save_img = Image.fromarray(color_pred)
#                 save_img_name = os.path.basename(file_name[i])  # Get only the file name
#                 save_img_path = os.path.join(args.save_path, f'{save_img_name}.png')
#                 save_img.save(save_img_path)

#             data_list.append([mask.numpy().flatten(), pred.flatten()])
#         filename = os.path.join(args.save_path, 'result.txt')
#         get_iou(data_list, NUM_CLASSES[args.dataset], filename, DATASET)

# def get_iou(data_list, class_num, save_path=None, dataset_name=None):
#     from multiprocessing import Pool
#     from utils.metric import ConfusionMatrix

#     ConfM = ConfusionMatrix(class_num)
#     f = ConfM.generateM
#     pool = Pool()
#     m_list = pool.map(f, data_list)
#     pool.close()
#     pool.join()

#     for m in m_list:
#         ConfM.addM(m)

#     aveJ, j_list, M = ConfM.jaccard()

#     if dataset_name == 'MSL' or dataset_name == 'MER':
#         classes, _ = MARS()
#     elif dataset_name == 'iSAID':
#         classes, _ = iSAID()
#     elif dataset_name == 'GID-15':
#         classes, _ = GID15()
#     elif dataset_name == 'Vaihingen':
#         classes, _ = Vaihingen()
#     elif dataset_name == 'DFC22':
#         classes, _ = DFC22()

#     for i, iou in enumerate(j_list):
#         print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i] * 100))

#     print('meanIOU {:.2f}'.format(aveJ * 100) + '\n')
#     if save_path:
#         with open(save_path, 'w') as f:
#             for i, iou in enumerate(j_list):
#                 f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i] * 100) + '\n')
#             f.write('meanIOU {:.2f}'.format(aveJ * 100) + '\n')

# if __name__ == '__main__':
#     args = parse_args()

#     if args.data_root is None:
#         args.data_root = {'GID-15': GID15_DATASET_PATH,
#                           'iSAID': iSAID_DATASET_PATH,
#                           'MER': MER_DATASET_PATH,
#                           'MSL': MSL_DATASET_PATH,
#                           'Vaihingen': Vaihingen_DATASET_PATH,
#                           'DFC22': DFC22_DATASET_PATH}[args.dataset]

#     print(args)
#     main(args)
from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from utils.dataset_name import *
from utils.utils import count_params, meanIOU, color_map

import argparse
from copy import deepcopy
import numpy as np
import os
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

NUM_CLASSES = {'DFC22': 12, 'iSAID': 15, 'MER': 9, 'MSL': 9, 'Vaihingen': 5, 'GID-15': 15}
DATASET = 'DFC22'
WEIGHTS = '/data5/WSCL_depth/exp_depth/DFC22/1-4_20_1/SDC/deeplabv2_resnet101_38.28.pth'

DFC22_DATASET_PATH = '/data5/WSCL_depth/dataset/splits/DFC22/'
iSAID_DATASET_PATH = 'Your local path'
MER_DATASET_PATH = 'Your local path'
MSL_DATASET_PATH = 'Your local path'
Vaihingen_DATASET_PATH = 'Your local path'
GID15_DATASET_PATH = 'Your local path'

def parse_args():
    parser = argparse.ArgumentParser(description='WSCL Framework')

    # Basic settings
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['DFC22', 'iSAID', 'MER', 'MSL', 'Vaihingen', 'GID-15'],
                        default=DATASET)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv2')
    args = parser.parse_args()
    return args

def DFC22():
    class_names = ['Urban fabric', 'Industrial', 'Mine', 'Artificial', 'Arable', 'Permanent crops',
                   'Pastures', 'Forests', 'Herbaceous', 'Open spaces', 'Wetlands', 'Water']
    return class_names

def main(args):
    model = DeepLabV2(args.backbone, NUM_CLASSES[args.dataset])
    model.load_state_dict(torch.load(WEIGHTS))
    model = DataParallel(model).cuda()

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=8,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    eval(model, valloader, valset, args)

def eval(model, valloader, valset, args):
    model.eval()
    tbar = tqdm(valloader)
    data_list = []

    with torch.no_grad():
        for idx, (img, mask, depth_img, file_name) in enumerate(tbar):
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu().numpy()

            data_list.append([mask.numpy().flatten(), pred.flatten()])
        
        # 计算 IoU 指标，不再保存 result.txt 文件
        get_iou(data_list, NUM_CLASSES[args.dataset], dataset_name=args.dataset)

# 调整 get_iou 函数的调用
def get_iou(data_list, class_num, dataset_name=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = DFC22() if dataset_name == 'DFC22' else None  # 根据需要调整其他数据集的类名

    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], j_list[i] * 100))

    print('meanIOU {:.2f}'.format(aveJ * 100) + '\n')


if __name__ == '__main__':
    args = parse_args()

    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    print(args)
    main(args)
