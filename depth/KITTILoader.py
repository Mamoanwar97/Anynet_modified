import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)

def disparity_npy_loader(path):
    return np.load(path).astype(np.float32)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader, load_npy=False):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.load_npy = load_npy

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.load_npy:
            dataL = disparity_npy_loader(disp_L)
        else:
            dataL = self.dploader(disp_L)
        # print(" a7la 3esh taza 3alyk")
        # print(np.min(dataL), np.max(dataL))

        if self.training:  
           w, h = left_img.size
           th, tw = 256, 512
 
           x1 = random.randint(0, w - tw)
           y1 = random.randint(0, h - th)

           left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
           right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

           if not self.load_npy:
                if np.max(dataL) <= 256:
                    dataL = np.ascontiguousarray(dataL, dtype=np.float32)
                else:
                    dataL = np.ascontiguousarray(dataL, dtype=np.float32)/256

           dataL = dataL[y1:y1 + th, x1:x1 + tw]
           dataL = (0.54 * 7.215377000000e+02) / dataL

           processed = preprocess.get_transform(augment=False)  
           left_img   = processed(left_img)
           right_img  = processed(right_img)

           if self.load_npy:
               dataL = torch.from_numpy(dataL).float()
           return left_img, right_img, dataL
        else:
           w, h = left_img.size

           left_img = left_img.crop((w - 1200, h - 352, w, h))
           right_img = right_img.crop((w - 1200, h - 352, w, h))
           w1, h1 = left_img.size

        #    print("Before processing")
        #    print(np.min(dataL), np.max(dataL))
           if not self.load_npy:
                if np.max(dataL) <= 256:
                    dataL = np.ascontiguousarray(dataL, dtype=np.float32)
                else:
                    dataL = np.ascontiguousarray(dataL, dtype=np.float32)/256

           dataL = dataL[h - 352:h, w - 1200:w]
           dataL = (0.54 * 7.215377000000e+02) / dataL

           processed = preprocess.get_transform(augment=False)  
           left_img       = processed(left_img)
           right_img      = processed(right_img)
        #    print("After processing")
        #    print(np.min(dataL), np.max(dataL))
        #    print("\n\n")
           if self.load_npy:
               dataL = torch.from_numpy(dataL).float()
           return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
