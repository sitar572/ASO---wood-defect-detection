import os
import cv2
from tqdm import tqdm
from PIL import Image
from dataset.params import Params


def create_class_dirs():
    # Create class directories for each label
    for label in Params.labels:
        label_path = os.path.join(Params.proc_imgs_path, label)
        os.mkdir(label_path)
    print('Folders created.')


def get_defect(line, width, height):
    # Get bounding box and label from file
    line = line.replace(',', '.')
    splitted_line = line.split("\t")

    label = splitted_line[0]
    min_x = int(float(splitted_line[1]) * width)
    min_y = int((float(splitted_line[2])) * height)
    max_x = int(float(splitted_line[3]) * width)
    max_y = int(float(splitted_line[4]) * height)

    return label, min_x, min_y, max_x, max_y


def crop_defects():
    print('Processing files:')

    images_files = os.listdir(Params.raw_imgs_path)

    # Labels samples counters
    counters = dict((label, 0) for label in Params.labels)

    for file_name in tqdm(images_files):
        file_path = os.path.join(Params.raw_imgs_path, file_name)

        # Find bounding boxes file
        splitted_img_name = file_name.split('.')
        img_name = splitted_img_name[0]
        bounding_path = os.path.join(
            Params.boundings_path, img_name + '_anno.txt')

        # Open image
        img = Image.open(file_path)
        width, height = img.size

        # Open bounding boxes file
        bounding_file = open(bounding_path, 'r')
        boundings_lines = bounding_file.readlines()

        # Crop defects
        for line in boundings_lines:
            # Get defect size
            label, min_x, min_y, max_x, max_y = get_defect(line, width, height)

            # Crop and save
            cropped_img = img.crop((min_x, min_y, max_x, max_y))

            cropped_width, cropped_height = cropped_img.size
            if cropped_width > Params.min_cropped_width and \
                    cropped_height > Params.min_cropped_height:
                new_file_name = str(counters[label]) + '.jpg'
                counters[label] += 1
                cropped_img.save(os.path.join(
                    Params.proc_imgs_path, label, new_file_name))


def process_dataset():
    if not os.path.isdir(Params.proc_imgs_path):
        os.mkdir(Params.proc_imgs_path)
    create_class_dirs()
    crop_defects()


# Retruns dict label_path: number_of_samples
def get_data_counters():
    labels = os.listdir(Params.proc_imgs_path)
    data_counters = {}

    for label in labels:
        label_dir = os.path.join(Params.proc_imgs_path, label)
        samples_number = len(os.listdir(label_dir))

        if samples_number > 0:
            data_counters[label] = samples_number

    return data_counters


def annotate_defect(img, label, min_x, min_y, max_x, max_y):
    # Annotation style
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (100, 100, 100)
    thickness = 4

    # Create bounding on the image
    img = cv2.rectangle(
        img,
        (min_x, min_y),
        (max_x, max_y),
        color,
        thickness * 2)

    # Describe bounding
    text_cords = (max_x, int((min_y + max_y) / 2))

    img = cv2.putText(img, str(label), text_cords, font,
                      fontScale, color, thickness, cv2.LINE_AA)

    return img


def annotate_sample(name: str):
    img = cv2.imread(os.path.join(Params.samples_path, name + '.bmp'))
    height = img.shape[0]
    width = img.shape[1]

    # Open bounding boxes file
    bounding_file = open(os.path.join(
        Params.samples_path, name + '_anno.txt'), 'r')
    boundings_lines = bounding_file.readlines()

    for line in boundings_lines:
        # Get defect
        label, min_x, min_y, max_x, max_y = get_defect(line, width, height)
        img = annotate_defect(img, label, min_x, min_y, max_x, max_y)

    return img
