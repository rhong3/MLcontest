import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io
import os
from sklearn.cluster import KMeans
np.random.seed(1234)
import torch
from torch.autograd import Variable
from imageio import imread
import re


ROOT = "../inputs/stage_1_train"
ROOT_IMAGE_PATTERN = "%s/{}/images/{}.png" % ROOT
ROOT_IMAGEPAD_PATTERN = "%s/{}/images/{}_pad.png" % ROOT
ROOT_LABEL_PATTERN = "%s/{}/label/Combined.png" % ROOT
ROOT_LABELPAD_PATTERN = "%s/{}/label/Combined_pad.png" % ROOT
ROOTT = "../inputs/stage_1_test"
ROOTT_IMAGE_PATTERN = "%s/{}/images/{}.png" % ROOTT
ROOTT_IMAGEPAD_PATTERN = "%s/{}/images/{}_pad.png" % ROOTT
ROOTT_LABEL_PATTERN = "%s/{}/label/Combined.png" % ROOT
ROOTT_LABELPAD_PATTERN = "%s/{}/label/Combined_pad.png" % ROOTT


def read_lite(summary, mode, root, root_IMAGE_PATTERN, root_IMAGEPAD_PATTERN, root_LABEL_PATTERN, root_LABELPAD_PATTERN):
    sample = []
    for index, row in summary.iterrows():
        ls = []
        color = row['hsv_cluster']
        W = row['width']
        H = row['height']
        id = row['image_id']
        if color == 0:
            type = 'fluorescence'
        elif color == 1:
            type = 'histology'
        elif color == 2:
            type = 'light'
        image_path = root_IMAGE_PATTERN.format(id, id)
        if H != W:
            image_path = root_IMAGEPAD_PATTERN.format(id, id)
            label_path = root_LABELPAD_PATTERN.format(id)

        else:
            label_path = root_LABEL_PATTERN.format(id)
        ls.append(type)
        ls.append(image_path)
        if mode == 'train':
            ls.append(label_path)
        ls.append(W)
        ls.append(H)
        ls.append(id)
        sample.append(ls)
    if mode == 'train':
        df = pd.DataFrame(np.array(sample), columns=['Type', 'Image', 'Label', 'Width', 'Height', 'ID'])
    else:
        df = pd.DataFrame(np.array(sample), columns=['Type', 'Image', 'Width', 'Height', 'ID'])
    return df


train = pd.read_csv('../inputs/stage_1_train/summary.csv', header = 0)
trsample = read_lite(train, 'train', ROOT, ROOT_IMAGE_PATTERN, ROOT_IMAGEPAD_PATTERN, ROOT_LABEL_PATTERN, ROOT_LABELPAD_PATTERN)
trsample.to_csv('../inputs/stage_1_train/samples.csv', index = False, header = True)

test = pd.read_csv('../inputs/stage_1_test/summary.csv', header = 0)
tesample = read_lite(test, 'test', ROOTT, ROOTT_IMAGE_PATTERN, ROOTT_IMAGEPAD_PATTERN, ROOTT_LABEL_PATTERN, ROOTT_LABELPAD_PATTERN)
tesample.to_csv('../inputs/stage_1_test/samples.csv', index = False, header = True)