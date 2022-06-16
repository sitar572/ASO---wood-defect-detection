import os
import pickle
import cv2
from dataset.dataset_utils import process_dataset
from dataset.params import Params
from classification.wood_feature import datasets_features
from classification.model import fit_model, predict_sample
from utils import get_project_root


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

    # Test example
    # model = pickle.load(open(filename, 'rb'))

    act, pred = predict_sample('124400077', model)

    act = cv2.resize(act, (920, 400), interpolation=cv2.INTER_AREA)
    pred = cv2.resize(pred, (920, 400), interpolation=cv2.INTER_AREA)

    cv2.imshow('act', act)
    cv2.imshow('pred', pred)
    cv2.waitKey(-1)


if __name__ == '__main__':
    main()
