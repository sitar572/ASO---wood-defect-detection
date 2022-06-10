import os
from dataset.dataset_utils import process_dataset
from dataset.params import Params


def main():
    if not os.path.isdir(Params.proc_imgs_path):
        process_dataset()


if __name__ == '__main__':
    main()
