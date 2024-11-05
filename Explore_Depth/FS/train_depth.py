import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random, time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils_depth import *
from torch.autograd import Variable
from IPython.display import clear_output
from model.FTUNetFormer_1 import ft_unetformer as FTUNetFormer
from skimage.color import rgb2gray
from tqdm import tqdm
import tifffile
# import cv2
# import imageio
# import tifffile
# import numpy as np
# import torch
# from tqdm import tqdm
from model.UNetFormer import UNetFormer as UNetFormer
from model.ABCNet import ABCNet


def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()


try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
if MODEL == 'FTUNetformer':
    net = FTUNetFormer(num_classes=N_CLASSES).cuda()
elif MODEL == 'UNetformer':
    net = UNetFormer(num_classes=N_CLASSES).cuda()
elif MODEL == 'ABCNet':
    net = ABCNet(num_classes=N_CLASSES).cuda()
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print(params)

# Load the datasets
print("training : ", len(train_ids))
print("testing : ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=False)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

base_lr = 0.005
LBABDA_BDY = 0.1
LBABDA_OBJ = 1
print("LBABDA_BDY: ", LBABDA_BDY)
print("LBABDA_OBJ: ", LBABDA_OBJ)
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params':[value],'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [15,25,35], gamma=0.1)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    if DATASET == 'Loveda':
        eroded_labels = ((np.asarray(io.imread(ERODED_FOLDER.format(id)), dtype='int64') - 1) for id in test_ids)
    else:
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    # Switch the network to inference mode
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
                outs = net(image_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            clear_output()

    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train_and_test(net, optimizer, epochs, scheduler=scheduler, weights=WEIGHTS, save_epoch=1, start_testing_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()
    iter_ = 0
    MIoU_best = 0
    criterion = DepthAwareSegmentationLoss(lambda_edge=1.0, lambda_consistency=0.5).cuda()
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        for batch_idx, (data, boundary, object, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss_ce_1 =loss_calc(output, target,weights)
            # DAE LOSS
            loss_222 = criterion(output,  object)
            loss = loss_ce_1 + loss_222
            loss.backward()
            optimizer.step()
            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])
            if iter_ % 1 == 0:
                clear_output()
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_ce: {:.6f}\tLoss_depth: {:.6f}\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss_ce_1.data, loss_222.data,  loss.data, accuracy(pred, gt)))
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0 and e >= start_testing_epoch:
            # We validate with the largest possible stride for faster computing
            net.eval()
            MIoU = test(net, test_ids, all=False, stride=Stride_Size)
            net.train()
            if MIoU > MIoU_best:
                if DATASET == 'Vaihingen':
                    torch.save(net.state_dict(), '/data5/open_code/resulr_shendu/unetformer_depth_11/epoch{}_{}'.format(e, MIoU))
                elif DATASET == 'Loveda':
                    torch.save(net.state_dict(), './Result/loveda/0.25/1/Loveda_PL_gen_model')
                elif DATASET == 'Potsdam':
                    torch.save(net.state_dict(), '/data5/open_code/resulr_shendu/potsdam/CMTF_YUANSHI/epoch{}_{}'.format(e, MIoU))

                elif DATASET == 'DFC22':
                    torch.save(net.state_dict(),
                               '/data5/open_code/resulr_shendu/DFC22/FT_yuanshi/epoch{}_{}'.format(e,
                                                                                                                  MIoU))
                elif DATASET == 'MER':
                    torch.save(net.state_dict(),
                               '/data5/open_code/resulr_shendu/MER/FT_yuanshi/epoch{}_{}'.format(e,
                                                                                                                 MIoU))
                elif DATASET == 'MSL':
                    torch.save(net.state_dict(),
                               '/data5/open_code/resulr_shendu/MSL/unetformer_depth/epoch{}_{}'.format(e,
                                                                                                                 MIoU))
                elif DATASET == 'GID-15':
                    torch.save(net.state_dict(),
                               '/data5/open_code/resulr_shendu/GID-15/FT_depth/epoch{}_{}'.format(e,
                                                                                                                 MIoU))
                elif DATASET == 'iSAID':
                    torch.save(net.state_dict(),
                               '/data5/open_code/resulr_shendu/iSAID/unetformer_depth/epoch{}_{}'.format(e,
                                                                                                                 MIoU))

                MIoU_best = MIoU

if MODE == 'Train':
    train_and_test(net, optimizer, 50, scheduler)

elif MODE == 'Test':
    # Perform testing here after training
    if DATASET == 'GID-15':
        net.load_state_dict(torch.load('/data5/open_code/resulr_shendu/GID-15/CMTF_depth/epoch5_0.5956591151096278')) # Replace YOUR_MODEL with actual model path
        print("load success")
        net.eval()
        MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=128)
        print("MIoU: ", MIoU)
        # for p, id_ in zip(all_preds, test_ids):
        #     img = convert_to_color(p)
        #     io.imsave('/data5/open_code/test/ssl/LSST/0.25/depth/{}.png'.format(id_), img)