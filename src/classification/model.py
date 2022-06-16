import os
import cv2
from sklearn import svm
from dataset.dataset_utils import annotate_sample, annotate_defect
from classification.segmentation import crop_wood, bound_defects
from classification.wood_feature import features_vector
from dataset.params import Params


def fit_model(train_data, test_data, train_labels, test_labels):
    clf = svm.SVC(decision_function_shape='ovo')

    print('Model fitting.')
    clf.fit(train_data, train_labels)

    print('Testing.')
    result = clf.score(test_data, test_labels)

    print('Accuracy:')
    print(result)

    return clf


def predict_sample(name: str, model):
    # Annotate sample with actual labels
    annotated_sample = annotate_sample(name)

    # Read sample
    sample = cv2.imread(os.path.join(Params.samples_path, name + '.bmp'))

    # Remove background
    img = crop_wood(sample)

    # Get defects boundings
    height = img.shape[0]
    width = img.shape[1]
    boundings = bound_defects(img)

    # Predict defects
    for bounding in boundings:
        # Crop defect
        x, y, w, h = bounding
        x = int(x * width)
        y = int(y * height)
        w = int(w * width)
        h = int(h * height)
        cropped_img = img[y:y+h, x:x+w]

        # Get feature vector and predicton
        fv = features_vector(cropped_img)
        predicted_label = model.predict(fv.reshape(1, -1))

        # Annotate defect with predicted label
        img = annotate_defect(img, predicted_label[0], x, y, x+w, y+h)

    return annotated_sample, img
