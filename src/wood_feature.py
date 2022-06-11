import os
import cv2
import numpy as np
from skimage import feature
from tqdm import tqdm


def hog_features(image):
    fv = feature.hog(image, orientations=8,
                     pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1),
                     visualize=False,
                     feature_vector=True)
    return fv


def lbp_features(image):
    lbp = feature.local_binary_pattern(image, 8, 3, 'default')
    lbp = np.float32(lbp)
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    lbp_hist = lbp_hist.flatten()
    return lbp_hist


def features_vector(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (100, 100))

    hf = hog_features(image)
    lf = lbp_features(image)
    cat_features = np.concatenate((hf, lf))

    return cat_features


def extract_set_features(data_root, data_counters):
    features = []
    labels = []

    for label in data_counters:
        print('Extracting features of \'' + label + '\' samples:')
        for index in tqdm(range(data_counters[label])):
            image = cv2.imread(os.path.join(
                data_root,
                label,
                str(index) + '.jpg'))

            fv = features_vector(image)

            features.append(fv)
            labels.append(label)

    return features, labels
