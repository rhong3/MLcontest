import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(1234)
import torch
from torch.autograd import Variable
from imageio import imread
from torch.nn import functional as F
from torch.nn import init


# Use cuda or not
USE_CUDA = 1

class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 1, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x

# if __name__ == '__main__':
#     net = UNet()
#     print(net)
#
#     test_x = Variable(torch.FloatTensor(1, 3, 1024, 1024))
#     out_x = net(test_x)
#
#     print(out_x.size())

# Initial weights
def init_weights(module):
    for name, param in module.named_parameters():
        if name.find('weight') != -1:
            if len(param.size()) == 1:
                Cuda(init.uniform(param.data, 1).type(torch.DoubleTensor))
            else:
                Cuda(init.xavier_uniform(param.data).type(torch.DoubleTensor))
        elif name.find('bias') != -1:
            Cuda(init.constant(param.data, 0).type(torch.DoubleTensor))

def Cuda(obj):
    if USE_CUDA:
        if isinstance(obj, tuple):
            return tuple(cuda(o) for o in obj)
        elif isinstance(obj, list):
            return list(cuda(o) for o in obj)
        elif hasattr(obj, 'cuda'):
            return obj.cuda()
    return obj


def reader (list):
    labellist = []
    imlist = []
    for index, row in list.iterrows():
        name = row['Image']
        im = imread(name)
        for i in range(3):
            im[:, :, i] = (im[:, :, i] - np.mean(im[:, :, i])) / np.std(im[:, :, i])
        im = np.reshape(im, [3, im.shape[0], im.shape[1]])
        imlist.append(im)
        lname = row['Label']
        la = imread(lname)
        la = np.reshape(la, [1, la.shape[0], la.shape[1]])
        labellist.append(la)
    imlist = np.array(imlist)
    labellist = np.array(labellist)
    return imlist, labellist

def dice_loss(y_pred, target):
    pred_vec = y_pred.view(-1)
    target_vec = Cuda(target.view(-1).type(torch.ByteTensor))
    iou = np.linspace(0.5, 0.95, 10)
    ppv = []
    for i in iou:
        tp = (pred_vec > 0.5) * target_vec.sum()
        pred = pred_vec > 0.5
        ppv.append(tp / (pred.sum() + target_vec.sum() - tp))
    ave_ppv = np.mean(ppv)
    return ave_ppv


def train(bs, sample, vasample, ep, ilr):
    batch_size = bs
    grad_accu_times = 8
    init_lr = ilr
    # img_csv_file = 'train_masks.csv'
    # train_img_dir = 'train'
    # train_mask_dir = 'train_masks_png'

    model = Cuda(UNet())
    init_weights(model)
    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    opt = torch.optim.RMSprop(model.parameters(), lr=init_lr)
    opt.zero_grad()
    rows_trn = sample.shape[0]
    batches_per_epoch = rows_trn // bs

    # forward_times = 0
    for epoch in range(ep):
        lr = init_lr * (0.1 ** (epoch // 10))
        order = np.arange(rows_trn)
        for itr in range(batches_per_epoch):
            rows = order[itr * bs: (itr + 1) * bs]
            if itr + 1 == batches_per_epoch:
                rows = order[itr * bs:]
            # read in a batch
            trim, trla = reader(sample.loc[rows[0]:rows[-1], :])
            if USE_CUDA:
                x = Cuda(Variable(torch.from_numpy(trim).type(torch.FloatTensor)))
                y = Cuda(Variable(torch.from_numpy(trla).type(torch.FloatTensor)))
            else:
                x = Variable(torch.from_numpy(trim).type(torch.FloatTensor))
                y = Variable(torch.from_numpy(trla).type(torch.FloatTensor))
        pred_mask = model(x)
        # vaim, vala = reader(vasample)
        vlosslist = []
        for itr in range(vasample.shape[0]):
            vaim, vala = reader(vasample.loc[itr:itr,:])
            if USE_CUDA:
                xv = Cuda(Variable(torch.from_numpy(vaim).type(torch.FloatTensor)))
                yv = Cuda(Variable(torch.from_numpy(vala).type(torch.FloatTensor)))
            else:
                xv = Variable(torch.from_numpy(vaim).type(torch.FloatTensor))
                yv = Variable(torch.from_numpy(vala).type(torch.FloatTensor))
            pred_maskv = model(xv)
            vloss = loss_fn(pred_maskv, yv)
            vloss += dice_loss(F.sigmoid(pred_maskv), yv)
            vlosslist.append(vloss.cpu().data.numpy()[0])
        loss = loss_fn(pred_mask, y)
        loss += dice_loss(F.sigmoid(pred_mask), y)
        loss.backward()
        vlossa = np.mean(vlosslist)
        print('Epoch {:>3} |lr {:>1.5f} | Loss {:>1.5f} | VLoss {:>1.5f} '.format(epoch + 1, lr, loss.cpu().data.numpy()[0], vlossa))
        opt.step()
        opt.zero_grad()

        for param_group in opt.param_groups:
            param_group['lr'] = lr


        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
            }
            torch.save(checkpoint, 'unet_test-{}'.format(epoch+1))


trsample = pd.read_csv('../inputs/stage_1_train/trsamples.csv', header = 0, usecols=['Image', 'Label'])
vasample = pd.read_csv('../inputs/stage_1_train/vasamples.csv', header = 0, usecols=['Image', 'Label'])
train(1, trsample, vasample, 10, 0.01)


# # Get train and test IDs
# train_ids = next(os.walk(TRAIN_PATH))[1]
# test_ids = next(os.walk(TEST_PATH))[1]
#
# # Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
# def rle_encoding(x):
#     dots = np.where(x.T.flatten() == 1)[0]
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if (b>prev+1): run_lengths.extend((b + 1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths
#
# def prob_to_rles(x, cutoff=0.5):
#     lab_img = label(x > cutoff)
#     for i in range(1, lab_img.max() + 1):
#         yield rle_encoding(lab_img == i)
#
#
# new_test_ids = []
# rles = []
# for n, id_ in enumerate(test_ids):
#     rle = list(prob_to_rles(preds_test_upsampled[n]))
#     rles.extend(rle)
#     new_test_ids.extend([id_] * len(rle))
#
#
#
# # Create submission DataFrame
# sub = pd.DataFrame()
# sub['ImageId'] = new_test_ids
# sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
# sub.to_csv('sub-dsbowl2018-1.csv', index=False)