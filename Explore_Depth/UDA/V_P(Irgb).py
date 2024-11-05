import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
import torch.utils.data as data
import torch.optim as optim
import torch.nn.init
from torch.autograd import Variable
from IPython.display import clear_output
import os
import time
from utils import *
from transdiscri import transDiscri
from FTUNetFormer_11 import ft_unetformer as ViT_seg
from func import loss_calc, bce_loss
from loss import entropy_loss
from func import prob_2_entropy
import torch.backends.cudnn as cudnn
import random
def seed_torch(seed=1034):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener


# Parameters

WINDOW_SIZE = (256, 256) # Patch size


STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "/data5/GLGAN/datasets/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

MAIN_FOLDER_V = FOLDER + 'postdam/'
DATA_FOLDER_V = MAIN_FOLDER_V + 'img_rgbir/top_potsdam_{}_RGBIR.tif'
LABEL_FOLDER_V = MAIN_FOLDER_V + 'label/top_potsdam_{}_label.tif'
Depth_FOLDER_V = MAIN_FOLDER_V + 'depth_3000/top_potsdam_{}_RGB.png'

MAIN_FOLDER_P = FOLDER + 'vaihingen/'
DATA_FOLDER_P = MAIN_FOLDER_P + 'img_irrg/top_mosaic_09cm_area{}.tif'
LABEL_FOLDER_P = MAIN_FOLDER_P + 'label/top_mosaic_09cm_area{}.tif'
Depth_FOLDER_P = MAIN_FOLDER_P + 'depth_1281/top_mosaic_09cm_area{}.png'
# net = ResUnetPlusPlus(3).cuda()

model = ViT_seg(num_classes=N_CLASSES)
params = 0
for name, param in model.named_parameters():
    params += param.nelement()
print(params)
# saved_state_dict = torch.load('2_Advent/pretrained/DeepLab_resnet_pretrained_imagenet.pth')
# new_params = model.state_dict().copy()
# for i in saved_state_dict:
#     i_parts = i.split('.')
#     if not i_parts[1] == 'layer5':
#         new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
# model.load_state_dict(new_params)
model.train()
model = model.cuda()
cudnn.benchmark = True
cudnn.enabled = True
d_aux = transDiscri(num_classes=N_CLASSES)
d_aux.train()
d_aux.cuda()

d_aux_depth = transDiscri(num_classes=N_CLASSES)
d_aux_depth.train()
d_aux_depth.cuda()


# seg maps, i.e. output, level
d_main = transDiscri(num_classes=N_CLASSES)
d_main.train()
d_main.cuda()


d_main_depth = transDiscri(num_classes=N_CLASSES)
d_main_depth.train()
d_main_depth.cuda()


train_ids_V = ['6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10', '3_12', '5_12', '7_11', '7_9', '6_9', '7_7',
             '4_12', '6_8', '6_12', '6_7', '4_11']
test_ids_V = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12']
train_ids_P = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
test_ids_P = ['5', '21', '15', '30']

print("Potsdam for training : ", train_ids_V)
print("Potsdam for testing : ", test_ids_V)
print("Vaihingen for training : ", train_ids_P)
print("Vaihingen for testing : ", test_ids_P)
DATASET_P = 'Vaihingen'
DATASET_V = 'Potsdam'
# train_set = ISPRS_dataset(train_ids_P, train_ids_V, DATASET_P, DATASET_V,
#                           DATA_FOLDER_P, DATA_FOLDER_V, LABEL_FOLDER_P, LABEL_FOLDER_V,
#                           Depth_FOLDER_P, Depth_FOLDER_V, cache=CACHE, RGB_flag=True)
train_set = ISPRS_dataset(train_ids_P, train_ids_V, DATASET_P, DATASET_V,
                          DATA_FOLDER_P, DATA_FOLDER_V, LABEL_FOLDER_P, LABEL_FOLDER_V,
                          Depth_FOLDER_P, Depth_FOLDER_V, cache=CACHE)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

print("Date OK!!!!")

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4
LAMBDA_ADV_MAIN = 0.001
LAMBDA_ADV_AUX = 0.0002
LAMBDA_DECOMP = 0.01
LAMBDA_SEG_MAIN = 1.0
LAMBDA_SEG_AUX = 0.1
print('LAMBDA_DECOMP: ', LAMBDA_DECOMP)
optimizer = optim.SGD(model.parameters(),
                        lr=LEARNING_RATE,
                        momentum=MOMENTUM,
                        weight_decay=WEIGHT_DECAY)

# discriminators' optimizers
optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=LEARNING_RATE_D,
                                betas=(0.9, 0.99))
optimizer_d_main = optim.Adam(d_main.parameters(), lr=LEARNING_RATE_D,
                                betas=(0.9, 0.99))
optimizer_d_aux_depth = optim.Adam(d_aux_depth.parameters(), lr=LEARNING_RATE_D,
                                betas=(0.9, 0.99))
optimizer_d_main_depth = optim.Adam(d_main_depth.parameters(), lr=LEARNING_RATE_D,
                                betas=(0.9, 0.99))

# labels for adversarial training
source_label = 0
target_label = 1


def test(test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER_V.format(id))[:, :, :3], dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER_V.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(LABEL_FOLDER_V.format(id))) for id in test_ids)

    # eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    model.eval()
    with torch.no_grad():
        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                        leave=False)):

                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                # Do the inference
                pred2, _, _, _= model(image_patches)

                # _,pred2, _, _= model(image_patches)

                outs = F.softmax(pred2, dim=1)
                outs = outs.data.cpu().numpy()

                # Fill in the results array-
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            clear_output()
            all_preds.append(pred)
            all_gts.append(gt_e)

            clear_output()
            # Compute some metrics
            # metrics(pred.ravel(), gt_e.ravel())
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    model.train()
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy









attention_mechanism1 = SFF(6).cuda()  # 对应pred1
attention_mechanism2 = SFF(6).cuda()  # 对应pred2


def train(epochs, start_save_epoch, save_epoch, weights=WEIGHTS):
    weights = weights.cuda()
    MIoU_best = 0.20
    iter = 0
    model.train()
    d_aux.train()
    d_main.train()
    d_aux_depth.train()
    d_main_depth.train()
    attention_mechanism1.train()
    attention_mechanism2.train()

    for epoch in range(1, epochs + 1):
        # print(f"开始第 {epoch}/{epochs} 轮训练")
        start_time = time.time()
        for batch_idx, (images, labels, depth_s, images_t, labels_t, depth_t)  in enumerate(train_loader):
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))

            optimizer_d_aux.zero_grad()
            optimizer_d_main.zero_grad()
            optimizer_d_aux_depth.zero_grad()
            optimizer_d_main_depth.zero_grad()

            adjust_learning_rate_D(optimizer_d_aux, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_d_main, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx, epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_d_aux_depth, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx,
                                   epochs * (10000 / BATCH_SIZE))
            adjust_learning_rate_D(optimizer_d_main_depth, (epoch - 1) * (10000 / BATCH_SIZE) + batch_idx,
                                   epochs * (10000 / BATCH_SIZE))

            #源域训练
            images = Variable(images).cuda()
            depth_s = Variable(depth_s).float().cuda()

            depth_s = depth_s.repeat(1, 2, 1, 1)  # 形状变为 [batch_size, 6, 256, 256]
        
            pred2, pred1, f_dix, f_dsx = model(images)
            loss_seg1 = loss_calc(pred1, labels, weights)
            loss_seg2 = loss_calc(pred2, labels, weights)
            
            

            fused_features_s1 = attention_mechanism1(pred1, depth_s)  # pred1与深度图融合
            fused_features_s2 = attention_mechanism2(pred2, depth_s)  # pred2与深度图融合

            # 目标域数据
            images_t = Variable(images_t).cuda()
            depth_t = Variable(depth_t).float().cuda()

            depth_t = depth_t.repeat(1, 2, 1, 1)  # 形状变为 [batch_size, 6, 256, 256]


            pred_target2, pred_target1, f_diy, f_dsy = model(images_t)
            
            # 融合目标域的深度图
            fused_features_t1 = attention_mechanism1(pred_target1, depth_t)
            fused_features_t2 = attention_mechanism2(pred_target2, depth_t)
            # print(f"目标域深度图融合完成")

            # 2. 判别未融合的分割图与深度图融合后的分割图
            D_out1_orig = d_aux(prob_2_entropy(F.softmax(pred_target1, dim=1)))
            D_out2_orig = d_main(prob_2_entropy(F.softmax(pred_target2, dim=1)))
            # print(f"未融合的分割图判别完成")

            D_out1_fused = d_aux_depth(prob_2_entropy(F.softmax(fused_features_t1, dim=1)))
            D_out2_fused = d_main_depth(prob_2_entropy(F.softmax(fused_features_t2, dim=1)))
            # print(f"融合后的分割图判别完成")

            loss_adv_target1_orig = bce_loss(D_out1_orig, source_label)
            loss_adv_target2_orig = bce_loss(D_out2_orig, source_label)

            loss_adv_target1_fused = bce_loss(D_out1_fused, source_label)
            loss_adv_target2_fused = bce_loss(D_out2_fused, source_label)
            # print(f"对抗损失计算完成")

            # discrepancy losses
            loss_base = multi_discrepancy(f_dix, f_diy)
            loss_detail = multi_discrepancy(f_dsx, f_dsy)
            # print(f"差异损失计算完成")


            # 总损失
            loss = (LAMBDA_SEG_MAIN * loss_seg2
                + LAMBDA_SEG_AUX * loss_seg1


                + LAMBDA_ADV_MAIN * (loss_adv_target2_orig + loss_adv_target2_fused )
                + LAMBDA_ADV_AUX * (loss_adv_target1_orig + loss_adv_target1_fused )


                + LAMBDA_DECOMP * (loss_base + loss_detail))
            loss.backward()
            # print(f"损失反向传播完成")

            # train D
            # bring back requires_grad
            for param in d_aux.parameters():
                param.requires_grad = True

            for param in d_main.parameters():
                param.requires_grad = True

            for param in d_aux_depth.parameters():
                param.requires_grad = True

            for param in d_main_depth.parameters():
                param.requires_grad = True

            # 1. 源域训练
            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            # 判别源域未融合的特征图
            D_out1_unfused_s = d_aux(prob_2_entropy(F.softmax(pred1, dim=1)))
            D_out2_unfused_s = d_main(prob_2_entropy(F.softmax(pred2, dim=1)))

            loss_D1_unfused_s = bce_loss(D_out1_unfused_s, source_label)
            loss_D2_unfused_s = bce_loss(D_out2_unfused_s, source_label)

            # 判别源域融合后的特征图
            fused_features_s1 = fused_features_s1.detach()
            fused_features_s2 = fused_features_s2.detach()

            D_out1_fused_s = d_aux_depth(prob_2_entropy(F.softmax(fused_features_s1, dim=1)))
            D_out2_fused_s = d_main_depth(prob_2_entropy(F.softmax(fused_features_s2, dim=1)))

            loss_D1_fused_s = bce_loss(D_out1_fused_s, source_label)
            loss_D2_fused_s = bce_loss(D_out2_fused_s, source_label)

            # 源域损失：未融合与融合后的损失相加
            
            
            loss_D1 = (loss_D1_unfused_s + loss_D1_fused_s) / 2
            loss_D2 = (loss_D2_unfused_s + loss_D2_fused_s ) / 2

            # 反向传播源域损失
            loss_D1.backward()
            loss_D2.backward()

            # 2. 目标域训练
            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            # 判别目标域未融合的特征图
            D_out1_unfused_t = d_aux(prob_2_entropy(F.softmax(pred_target1, dim=1)))
            D_out2_unfused_t = d_main(prob_2_entropy(F.softmax(pred_target2, dim=1)))
            loss_D1_unfused_t = bce_loss(D_out1_unfused_t, target_label)
            loss_D2_unfused_t = bce_loss(D_out2_unfused_t, target_label)

            # 判别目标域融合后的特征图
            fused_features_t1 = fused_features_t1.detach()
            fused_features_t2 = fused_features_t2.detach()
            D_out1_fused_t = d_aux_depth(prob_2_entropy(F.softmax(fused_features_t1, dim=1)))
            D_out2_fused_t = d_main_depth(prob_2_entropy(F.softmax(fused_features_t2, dim=1)))
            loss_D1_fused_t = bce_loss(D_out1_fused_t, target_label)
            loss_D2_fused_t = bce_loss(D_out2_fused_t, target_label)

            # 目标域损失：未融合与融合后的损失相加
            loss_D1 = (loss_D1_unfused_t + loss_D1_fused_t) / 2
            loss_D2 = (loss_D2_unfused_t + loss_D2_fused_t) / 2


            # 反向传播目标域损失
            loss_D1.backward()
            loss_D2.backward()


            optimizer.step()
            optimizer_d_aux.step()
            optimizer_d_main.step()


            if iter % 100 == 0:
                clear_output()
                pred = np.argmax(pred_target2.data.cpu().numpy()[0], axis=0)
                gt = labels_t.data.cpu().numpy()[0]
                end_time = time.time()
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)] lr: {:.12f} lr_D: {:.12f} Loss: {:.6f} Loss_Base: {:.6f} Loss_Detail: {:.6f} Loss_D1: {:.6f} Loss_D2: {:.6f} Accuracy: {:.2f}% Timeuse: {:.2f}'.format(
                    epoch, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'], optimizer_d_aux.state_dict()['param_groups'][0]['lr'],
                    loss_seg2.data, loss_base.data, loss_detail.data, loss_D1.data, loss_D2.data, accuracy(pred, gt), end_time - start_time))
                start_time = time.time()
            iter += 1
            del (images, labels, images_t, labels_t, loss, loss_base, loss_detail, loss_D1, loss_D2)

        if epoch >= start_save_epoch and epoch % save_epoch == 0:
            start_time = time.time()
            MIoU = test(test_ids_V, all=False, stride=128)
            end_time = time.time()
            print('Test Stide_32 time use: ', end_time - start_time)
            start_time = time.time()
            if MIoU > MIoU_best:
                torch.save(model.state_dict(), '/data5/GLGAN/result_xiugai/V_P(IRGB)/CGA_baoliu_pred1/V_P(IRGB)_epoch{}_{}'.format(epoch, MIoU))
                MIoU_best = MIoU
    print("Train Done!!")

train(100, 1, 1)





# #####   test   ####
# model.load_state_dict(torch.load('/data5/GLGAN/result_xiugai/V_P(IRGB)/CGA_D/V_P(IRGB)_epoch76_0.47332879896000096'))
# acc, all_preds, all_gts = test(test_ids_V, all=True, stride=128)
# print("Acc: ", acc)

# for p, id_ in zip(all_preds, test_ids_V):
#    img = convert_to_color(p)
#    io.imsave('./Test_Vision/PRGB2V_' + str(acc) + '_tile_{}.png'.format(id_), img)
