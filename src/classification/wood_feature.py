import os
import cv2
import numpy as np
from skimage import feature
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset.dataset_utils import get_data_counters
from dataset.params import Params


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

    # Exctract features of every image of every label
    for label in data_counters:
        print('Extracting features of \'' + label + '\' samples:')

        # For each image in label directory
        for index in tqdm(range(data_counters[label])):
            image = cv2.imread(os.path.join(
                data_root,
                label,
                str(index) + '.jpg'))

            fv = features_vector(image)

            features.append(fv)
            labels.append(label)

    return features, labels


def datasets_features():
    # Get not empty classes from processed dataset
    data_counters = get_data_counters()

    # Extracting features of dataset
    features, labels = extract_set_features(
        Params.proc_imgs_path, data_counters)

    features = np.array(features)
    labels = np.array(labels)

    print('Splitting dataset.')
    (train_data, test_data, train_labels, test_labels) = \
        train_test_split(features,
                         labels,
                         test_size=0.3,
                         random_state=5)

    return train_data, test_data, train_labels, test_labels
