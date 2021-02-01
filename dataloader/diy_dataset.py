import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, train_spit=None, val_list=None):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'

    if not train_spit is None:
        with open(train_spit) as f:
            trainlist = ([(str(x.strip()) + '.png') for x in f.readlines() if len(x) > 0])
        train = trainlist
    else:
        train = [x for x in os.listdir(os.path.join(filepath, 'training', left_fold)) if is_image_file(x)]
    
    left_train = [os.path.join(filepath, 'training', left_fold, img) for img in train]
    right_train = [os.path.join(filepath, 'training', right_fold, img) for img in train]
    left_train_disp = [os.path.join(filepath, 'training', disp_L, img) for img in train]


    if not val_list is None:
        with open(val_list) as f:
            vallist = ([(str(x.strip()) + '.png') for x in f.readlines() if len(x) > 0])
        val = vallist[:1000]
    else:
        val = [x for x in os.listdir(os.path.join(filepath, 'training', left_fold)) if is_image_file(x)]

    left_val = [os.path.join(filepath, 'training', left_fold, img) for img in val]
    right_val = [os.path.join(filepath, 'training', right_fold, img) for img in val]
    left_val_disp = [os.path.join(filepath, 'training', disp_L, img) for img in val]

    return left_train, right_train, left_train_disp, left_val, right_val, left_val_disp

def testloader(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'

    # all_index = np.arange(4)
    # np.random.shuffle(all_index)
    disp_files = [x for x in os.listdir(os.path.join(filepath, 'training', disp_L))]
    val = disp_files[:10]

    # val = [x for x in os.listdir(os.path.join(filepath, 'validation', left_fold)) if is_image_file(x)]
    left_val = [os.path.join(filepath, 'training', left_fold, img) for img in val]
    right_val = [os.path.join(filepath, 'training', right_fold, img) for img in val]
    left_val_disp = [os.path.join(filepath, 'training', disp_L, img) for img in val]
    return left_val, right_val, left_val_disp
