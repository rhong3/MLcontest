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
import os
import shutil


file = pd.read_csv('input/stage1_test_support/testset_summary.csv', header=0)

os.mkdir('input/stage_1_test/fluorescence')
os.mkdir('input/stage_1_test/histology')
os.mkdir('input/stage_1_test/light')


for index, row in file.iterrows():
    if row['hsv_cluster'] == 0:
        name = str(row['image_id'])
        shutil.copytree('input/stage1_test/'+name, 'input/stage_1_test/fluorescence/'+name)
    if row['hsv_cluster'] == 1:
        name = str(row['image_id'])
        shutil.copytree('input/stage1_test/'+name, 'input/stage_1_test/histology/'+name)
    if row['hsv_cluster'] == 2:
        name = str(row['image_id'])
        shutil.copytree('input/stage1_test/'+name, 'input/stage_1_test/light/'+name)
