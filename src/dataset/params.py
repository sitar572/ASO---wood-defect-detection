import os
from utils import get_project_root


class Params:
    # Labels list
    labels = ['Live_Knot',
              'Dead_Knot',
              'Knot_missing',
              'knot_with_crack',
              'Crack',
              'Quartzity',
              'resin',
              'Marrow',
              'Blue_stain',
              'overgrown'
              ]

    # Directories
    root = get_project_root()
    proc_imgs_path = os.path.join(root, 'data/dataset/processed')
    raw_imgs_path = os.path.join(root, 'data/dataset/raw/images')
    boundings_path = os.path.join(root, 'data/dataset/raw/boundings')

    # Images processing params
    min_cropped_width = 20
    min_cropped_height = 20
