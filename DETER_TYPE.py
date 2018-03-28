import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from textwrap import wrap
np.random.seed(1234)
import scipy.misc


STAGE1_TRAIN = "input/stage1_test"
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN

def get_domimant_colors(img, top_colors=2):
    img_l = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    clt = KMeans(n_clusters = top_colors)
    clt.fit(img_l)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_, hist

def deter_image(image_path, space="hsv"):
    image = skimage.io.imread(image_path)
    # Drop alpha which is not used
    image = image[:, :, :3]
    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(image, top_colors=1)
    dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] * dominant_colors_hsv.shape[1])
    a = dominant_colors_hsv.squeeze()
    if a[0] == 0 and a[1] == 0:
        if a[2] < 0.5:
            type = 'Fluorescence'
        else:
            type = 'Light'
    else:
        type = 'histology'
    return type


t = deter_image('input/stage_1_train/light/8d05fb18ee0cda107d56735cafa6197a31884e0a5092dc6d41760fb92ae23ab4/images/8d05fb18ee0cda107d56735cafa6197a31884e0a5092dc6d41760fb92ae23ab4.png')
print(t)
