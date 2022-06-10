import os
from tqdm import tqdm
from PIL import Image
from dataset.params import Params


def create_class_dirs():
    for label in Params.labels:
        label_path = os.path.join(Params.proc_imgs_path, label)
        os.mkdir(label_path)
    print('Folders created.')


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
            line = line.replace(',', '.')
            splitted_line = line.split("\t")

            label = splitted_line[0]
            min_x = int(float(splitted_line[1]) * width)
            min_y = int((float(splitted_line[2])) * height)
            max_x = int(float(splitted_line[3]) * width)
            max_y = int(float(splitted_line[4]) * height)

            # Crop and save
            cropped_img = img.crop((min_x, min_y, max_x, max_y))

            new_file_name = str(counters[label]) + '.jpg'
            counters[label] += 1

            cropped_width, cropped_height = cropped_img.size
            if cropped_width > Params.min_cropped_width and cropped_height > Params.min_cropped_height:
                cropped_img.save(os.path.join(
                    Params.proc_imgs_path, label, new_file_name))


def process_dataset():
    if not os.path.isdir(Params.proc_imgs_path):
        os.mkdir(Params.proc_imgs_path)
    create_class_dirs()
    crop_defects()
