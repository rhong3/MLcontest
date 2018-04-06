import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io
import os
from sklearn.cluster import KMeans
np.random.seed(1234)

ROOT = "inputs/stage_1_train"
ROOT_IMAGE_PATTERN = "%s/{}/images/{}.png" % ROOT
ROOT_IMAGEPAD_PATTERN = "%s/{}/images/{}_pad.png" % ROOT
ROOT_LABEL_PATTERN = "%s/{}/label/Combined.png" % ROOT
ROOT_LABELPAD_PATTERN = "%s/{}/label/Combined_pad.png" % ROOT
ROOTT = "inputs/stage_1_test"
ROOTT_IMAGE_PATTERN = "%s/{}/images/{}.png" % ROOTT
ROOTT_IMAGEPAD_PATTERN = "%s/{}/images/{}_pad.png" % ROOTT
ROOTT_LABEL_PATTERN = "%s/{}/label/Combined.png" % ROOT
ROOTT_LABELPAD_PATTERN = "%s/{}/label/Combined_pad.png" % ROOTT


###########################################################################################
def image_ids_in(root_dir, ignore=['.DS_Store', 'samples.csv','summary.csv', 'stage1_train_labels.csv', 'stage1_test_labels.csv', 'stage2_test_labels.csv']):
    ids = []
    for id in os.listdir(root_dir):
        if id in ignore:
            print('Skipping ID:', id)
        else:
            ids.append(id)
    return ids



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
    H, W, L = image.shape

    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    dominant_colors_hsv, dominant_rates_hsv = get_domimant_colors(image, top_colors=1)
    dominant_colors_hsv = dominant_colors_hsv.reshape(1, dominant_colors_hsv.shape[0] * dominant_colors_hsv.shape[1])
    a = dominant_colors_hsv.squeeze()
    if a[0] == 0 and a[1] == 0:
        if a[2] < 0.5:
            type = 'fluorescence'
        else:
            type = 'light'
    else:
        type = 'histology'
    return type, H, W, L


def read(root, root_IMAGE_PATTERN, root_IMAGEPAD_PATTERN, root_LABEL_PATTERN, root_LABELPAD_PATTERN):
    ids = image_ids_in(root)
    sample = []
    for id in ids:
        ls = []
        image_path = root_IMAGE_PATTERN.format(id, id)
        Type, H, W, L = deter_image(image_path)
        if H != W:
            image_path = root_IMAGEPAD_PATTERN.format(id, id)
            label_path = root_LABELPAD_PATTERN.format(id)

        else:
            label_path = root_LABEL_PATTERN.format(id)
        ls.append(Type)
        ls.append(image_path)
        ls.append(label_path)
        sample.append(ls)
    df = pd.DataFrame(np.array(sample), columns=['Type', 'Image', 'Label'])
    print(df)
    return df

# a = read(ROOT, ROOT_IMAGE_PATTERN, ROOT_IMAGEPAD_PATTERN, ROOT_LABEL_PATTERN, ROOT_LABELPAD_PATTERN)

