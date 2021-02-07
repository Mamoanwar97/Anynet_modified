import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, train_spit=None, val_list=None, load_npy=False):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    if load_npy:
        disp_L = 'disp_occ_0_npy/'
    else:
        disp_L = 'disp_occ_0/'

    if not train_spit is None:
        with open(train_spit) as f:
            trainlist = ([(str(x.strip())) for x in f.readlines() if len(x) > 0])
        train = trainlist
    else:
        train = [x[:-4] for x in os.listdir(os.path.join(filepath, 'training', left_fold)) if is_image_file(x)]
    
    left_train = [os.path.join(filepath, 'training', left_fold, img + '.png') for img in train]
    right_train = [os.path.join(filepath, 'training', right_fold, img + '.png') for img in train]
    if load_npy:
        left_train_disp = [os.path.join(filepath, 'training', disp_L, img + '.npy') for img in train]
    else:
        left_train_disp = [os.path.join(filepath, 'training', disp_L, img + '.png') for img in train]


    if not val_list is None:
        with open(val_list) as f:
            vallist = ([(str(x.strip())) for x in f.readlines() if len(x) > 0])
        val = vallist
    else:
        val = [x[:-4] for x in os.listdir(os.path.join(filepath, 'training', left_fold)) if is_image_file(x)]

    left_val = [os.path.join(filepath, 'training', left_fold, img+ '.png') for img in val]
    right_val = [os.path.join(filepath, 'training', right_fold, img+ '.png') for img in val]
    if load_npy:
        left_val_disp = [os.path.join(filepath, 'training', disp_L, img+ '.npy') for img in val]
    else:
        left_val_disp = [os.path.join(filepath, 'training', disp_L, img+ '.png') for img in val]

    return left_train, right_train, left_train_disp, left_val, right_val, left_val_disp

def testloader(filepath, limit=-1, split_file=None):
    left_fold = 'image_2/'
    right_fold = 'image_3/'

    if not split_file is None:
        with open(split_file) as f:
            test_files = ([(str(x.strip())+'.png') for x in f.readlines() if len(x) > 0])
    else:
        test_files = [x for x in os.listdir(os.path.join(filepath, 'training', right_fold))]

    if not limit == -1:
        test_files = test_files[:limit]

    left_val = [os.path.join(filepath, 'training', left_fold, img) for img in test_files]
    right_val = [os.path.join(filepath, 'training', right_fold, img) for img in test_files]
    return left_val, right_val
