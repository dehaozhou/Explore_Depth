from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils_j.utils import *
import argparse
import numpy as np
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile, clever_format
        
seed = 1234
set_random_seed(seed)

DATASET = 'Vaihingen'     # ['Vaihingen','DFC22', , 'MER', 'Potsdam'] ,'MSL''Loveda'
SPLIT = '1-8'     # ['1-4', '1-8', '100', '300']

# finish:
# DFC22: 1/4 1/8
# Vaihingen 1/4 1/8
# MER: 1/4 1/8
#POTSDAM: 1/4 1/8
# MSL:  1/4 1/8
# Loveda 1/4

DFC22_DATASET_PATH = '/data5/WMGS/dataset/splits/DFC22/'
iSAID_DATASET_PATH = '/data5/WMGS/dataset/splits/iSAID/'
MER_DATASET_PATH = '/data5/WMGS/dataset/splits/MER/'
MSL_DATASET_PATH = '/data5/WMGS/dataset/splits/MSL/'
Vaihingen_DATASET_PATH = '/data5/dehaozhou/WMGS/dataset/splits/Vaihingen/'
GID15_DATASET_PATH = '/data5/WMGS/dataset/splits/GID-15/'
Potsdam_DATASET_PATH = '/data5/WMGS/dataset/splits/Potsdam/'
Loveda_DATASET_PATH = '/data5/WMGS/dataset/splits/Loveda/'


PERCENT = 20
LAMBDA = 1
AUG_PROCESS_MODE = 'yes_seg_d_loss'



NUM_CLASSES = {'DFC22': 12, 'iSAID': 15, 'MER': 9, 'MSL': 9, 'Vaihingen': 5,'Potsdam': 5, 'GID-15': 15,'Loveda': 7}

def parse_args():
    parser = argparse.ArgumentParser(description='WMFSS Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['GID-15', 'iSAID', 'DFC22', 'MER', 'MSL', 'Vaihingen','Loveda','Potsdam'], default=DATASET)
    parser.add_argument('--percent', type=float, default=PERCENT, help='0~100, the low-entropy percent r')
    parser.add_argument('--lamb', type=int, default=LAMBDA, help='the trade-off weight to balance the supervised loss and the unsupervised loss')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv2')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/L.txt')
    parser.add_argument('--unlabeled-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/U.txt')
    # parser.add_argument('--depth_u-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/depth_u.txt')
    # parser.add_argument('--depth_l-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/depth_l.txt')
    # parser.add_argument('--sam_u-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/sam_u.txt')
    # parser.add_argument('--sam_l-id-path', type=str, default='./dataset/splits/' + DATASET + '/' + SPLIT + '/sam_l.txt')
    parser.add_argument('--save-path', type=str, default='./result_new/' + DATASET + '/' + SPLIT + '_' + str(PERCENT) + '_' + str(LAMBDA) + '/' + AUG_PROCESS_MODE)

    args = parser.parse_args()
    return args


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    create_path(args.save_path)

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=8, shuffle=False, pin_memory=False, num_workers=0, drop_last=False)

    trainset_u = SemiDataset(args.dataset, args.data_root, 'train_u', args.crop_size, args.unlabeled_id_path)
    trainset_l = SemiDataset(args.dataset, args.data_root, 'train_l', args.crop_size, args.labeled_id_path, nsample=len(trainset_u.ids))
    # depth_u = SemiDataset(args.dataset, args.data_root, 'depth_u', args.crop_size, args.depth_u_id_path, nsample=len(trainset_u.ids))
    # depth_l = SemiDataset(args.dataset, args.data_root, 'depth_l', args.crop_size, args.depth_l_id_path, nsample=len(trainset_l.ids))
    # sam_u = SemiDataset(args.dataset, args.data_root, 'sam_u', args.crop_size, args.depth_u_id_path,nsample=len(trainset_u.ids))
    # sam_l = SemiDataset(args.dataset, args.data_root, 'sam_l', args.crop_size, args.depth_u_id_path,nsample=len(trainset_l.ids))

    trainloader_u = DataLoader(trainset_u, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=0, drop_last=True)
    trainloader_l = DataLoader(trainset_l, batch_size=int(args.batch_size / 2), shuffle=True,
                               pin_memory=False, num_workers=0, drop_last=True)
    # trainloader_depth_u = DataLoader(depth_u, batch_size=int(args.batch_size / 2), shuffle=True,
    #                            pin_memory=False, num_workers=0, drop_last=True)
    # trainloader_depth_l = DataLoader(depth_l, batch_size=int(args.batch_size / 2), shuffle=True,
    #                            pin_memory=False, num_workers=0, drop_last=True)
    # trainloader_sam_u = DataLoader(sam_u, batch_size=int(args.batch_size / 2), shuffle=True,
    #                            pin_memory=False, num_workers=0, drop_last=True)
    # trainloader_sam_l = DataLoader(sam_l, batch_size=int(args.batch_size / 2), shuffle=True,
    #                            pin_memory=False, num_workers=0, drop_last=True)
    
    

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    # input_tensor = torch.randn(1, 3, args.crop_size, args.crop_size).cuda()  # 创建一个虚拟输入张量
    # model.eval()  # 切换到评估模式
    # with torch.no_grad():
    #     flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    #     flops, params = clever_format([flops, params], "%.3f")
    #     print(f"FLOPs: {flops}, Params: {params}")

    train(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, args)




def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, NUM_CLASSES[args.dataset])
    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #model = DataParallel(model).cuda() #多线程
    model = model.cuda(0) #单卡

    return model, optimizer


def train(model, trainloader_l, trainloader_u, valloader, criterion, optimizer, args):
    # 初始化训练参数
    iters = 0
    total_iters = len(trainloader_u) * args.epochs

    previous_best = 0.0
    previous_best_iou = 0.0
    weight_u = args.lamb

    N_CLASSES = NUM_CLASSES[DATASET]

    criterion_depth = Depth(lambda_seg_depth=1, lambda_smooth=0.5).cuda()


    depth_weight_map_generator = Depth_MoE(max_weight=3).cuda()

    WEIGHTS = torch.ones(N_CLASSES).cuda()


    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f  %s" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best, previous_best_iou))

        total_loss, total_loss_l, total_loss_u,total_depth_loss_l,total_depth_loss_u,total_loss_sam_u,total_loss_sam_l= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        tbar = tqdm(zip(trainloader_l, trainloader_u), total=len(trainloader_u))

        for i, ((img_l, mask_l,depth_l,sam_l), (img_u_w, img_u_s1, img_u_s2,depth_u,sam_u)) in enumerate(tbar):

            img_l, mask_l, depth_l,sam_l = img_l.cuda(), mask_l.cuda(), depth_l.cuda(),sam_l.cuda()
            img_u_w, img_u_s1, img_u_s2, depth_u,sam_u = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda(), depth_u.cuda(),sam_u.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_w = model(img_u_w)
                prob_u_w = pred_u_w.softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)



            rand_num = np.random.uniform(0, 1)  
            if rand_num < 0.5:
                conf_u_w, mask_u_w, img_u_s1,  depth_u, sam_u = generate_depth_sam(conf_u_w, mask_u_w, img_u_s1,
                                                                                          img_u_s2, depth_u,
                                                                                          sam_u)
            model.train()
            num_lb, num_ulb = img_l.shape[0], img_u_w.shape[0]
            preds = model(torch.cat((img_l, img_u_s1)))
            pred_l, pred_u_s = preds.split([num_lb, num_ulb])
            prob_u_s = pred_u_s.clone()
            em_u_s = entropy_map(prob_u_s.softmax(dim=1), 1)
            em_threshold = np.percentile(em_u_s.detach().cpu().numpy().flatten(), args.percent)
            loss_l = criterion(pred_l, mask_l)
            loss_u = criterion(pred_u_s, mask_u_w)
            loss_u = loss_u * (em_u_s <= em_threshold)
            loss_u = torch.mean(loss_u)
            weight_map_l = depth_weight_map_generator(depth_l,pred_l)  
            weight_map_u = depth_weight_map_generator(depth_u,pred_u_s)
            loss_depth_u = criterion_depth(pred_u_s, mask_u_w, depth_u, img_u_s1,WEIGHTS,weight_map_u)
            loss_depth_l = criterion_depth(pred_l, mask_l, depth_l, img_l,WEIGHTS,weight_map_l)
            weight_s = 1
            weight_b = 1

            loss = weight_s *(loss_l + loss_u) + weight_b*( loss_depth_l +  loss_depth_u)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            total_loss += loss.item()
            total_loss_l += loss_l.item()
            total_loss_u += loss_u.item()
            total_depth_loss_l += loss_depth_l.item()
            total_depth_loss_u += loss_depth_u.item()
            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
            tbar.set_description('Loss: %.6f, Loss_l: %.6f, Loss_u: %.6f, loss_depth_l: %.6f, loss_depth_u: %.6f' %
                                 (total_loss / (i + 1), total_loss_l / (i + 1), total_loss_u / (i + 1), total_depth_loss_l / (i + 1), total_depth_loss_u / (i + 1)))
        if (epoch + 1) % 1 == 0:
            metric = meanIOU(num_classes=NUM_CLASSES[args.dataset])
            model.eval()
            tbar = tqdm(valloader)
            with torch.no_grad():
                for img, mask, _ in tbar:
                    img = img.cuda()
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())
                    IOU, mIOU = metric.evaluate()
                    tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            mIOU *= 100.0
            IOU *= 100
            print('IoU: {}  | MIoU: {}'.format(IOU, mIOU))
            if mIOU > previous_best:
                if previous_best != 0:
                    os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
                previous_best = mIOU
                previous_best_iou = IOU
                torch.save(getattr(model, 'module', model).state_dict(),
                        os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

if __name__ == '__main__':
    args = parse_args()
    if args.epochs is None:
        args.epochs = {'GID-15': 200, 'iSAID': 200, 'MER': 100, 'MSL': 100, 'Vaihingen': 100, 'DFC22': 100,'Potsdam': 100,'Loveda': 100}[args.dataset]
    if args.lr is None:
        args.lr = {'GID-15': 0.001, 'iSAID': 0.001, 'MER': 0.001, 'MSL': 0.001,
                   'Vaihingen': 0.001,'Potsdam': 0.001,'Loveda': 0.001, 'DFC22': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'GID-15': 321, 'iSAID': 321, 'MER': 321, 'MSL': 321, 'Vaihingen': 321,'Potsdam': 321, 'DFC22': 321,'Loveda': 321}[args.dataset]
    if args.data_root is None:
        args.data_root = {'GID-15': GID15_DATASET_PATH,
                          'iSAID': iSAID_DATASET_PATH,
                          'MER': MER_DATASET_PATH,
                          'MSL': MSL_DATASET_PATH,
                          'Vaihingen': Vaihingen_DATASET_PATH,
                          'Potsdam': Potsdam_DATASET_PATH,
                          'Loveda': Loveda_DATASET_PATH,
                          'DFC22': DFC22_DATASET_PATH}[args.dataset]

    print(args)

    main(args)
