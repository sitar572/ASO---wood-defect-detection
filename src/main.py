import os
import pickle
import numpy as np
from dataset.dataset_utils import process_dataset, get_data_counters
from dataset.params import Params
from utils import get_project_root
from wood_feature import extract_set_features
from sklearn import svm
from sklearn.model_selection import train_test_split


def datasets_features():
    data_counters = get_data_counters()

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


def fit_model(train_data, test_data, train_labels, test_labels):
    clf = svm.SVC(decision_function_shape='ovo')

    print('Model fitting.')
    clf.fit(train_data, train_labels)

    print('Testing.')
    result = clf.score(test_data, test_labels)

    print('Accuracy:')
    print(result)

    return clf


def main():
    # Prepare dataset
    if not os.path.isdir(Params.proc_imgs_path):
        process_dataset()

    train_data, test_data, train_labels, test_labels = datasets_features()

    # Fit model
    model = fit_model(train_data, test_data, train_labels, test_labels)

    # Save the model to disk
    filename = os.path.join(get_project_root(), 'data/model/model.sav')
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    main()
