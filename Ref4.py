import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
rcParams["figure.figsize"] = 16, 6
plt.style.use("seaborn-white")

import os
from subprocess import check_output

import tensorflow as tf


PATH = "../input/"

print(check_output(["ls","../input/"]).decode("utf8"))

train1 = pd.read_csv(PATH + "stage1_train_labels.csv")
ss1 = pd.read_csv(PATH + "stage1_sample_submission.csv")


train1.head()

print("There are {} rows of data.".format(train1.shape[0]))

TARGET = "EncodedPixels"

ss1.head()

def dimg(idx):
    """
    Displays image corresponding to the id
    """
    img = mpimg.imread(PATH+"stage1_train/"+idx+"/"+"images/"+idx+".png")
    return img


def dmsk(idx):
    """
    Displays the masks corersponding to id
    """
    f = os.listdir(PATH + "stage1_train/" + idx + "/masks")[0]
    nim = mpimg.imread(PATH + "stage1_train/" + idx + "/masks/" + f)

    for m in os.listdir(PATH + "stage1_train/" + idx + "/masks")[1:]:
        nim += mpimg.imread(PATH + "stage1_train/" + idx + "/masks/" + m)

    return nim

def dbth(idx):
    """
    Display both the mask and the image
    """
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(dimg(idx))
    ax[1].imshow(dmsk(idx), cmap="Purples")
    plt.show()


for ind in train1.sample(5)["ImageId"].index:
    print("Image ID:", ind)
    dbth(train1.iloc[ind,0])


