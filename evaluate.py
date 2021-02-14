import numpy as np
import cupy as cp
import skimage
import skimage.io

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

from dataloader import KITTI_testloader as DA
from dataloader import test_dataset as ls
from preprocessing.kitti_util import Calibration
from preprocessing.generate_lidar import project_disp_to_points, Calibration
from preprocessing.kitti_sparsify import pto_ang_map

import argparse
from models.anynet import AnyNet
import time
import os
import sys
from tqdm.auto import tqdm
from multiprocessing import Process, Queue, Pool


parser = argparse.ArgumentParser(description='Evaluating Anynet')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--save_path', type=str, default='results/pseudoLidar_cupy/', help='the path of saving checkpoints and log')

""" OPTIONS """
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--threads', type=int, default=5)
parser.add_argument('--limit', type=int, default=-1)

""" Anynet modal args """
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')

""" LiDAR args """
parser.add_argument('--max_high', type=int, default=1)

""" Kitti sparsify args """
parser.add_argument('--H', default=64, type=int)
parser.add_argument('--W', default=512, type=int)
parser.add_argument('--D', default=700, type=int)
parser.add_argument('--slice', default=1, type=int)

args = parser.parse_args()

def main():
    global args

    test_left_img, test_right_img = ls.dataloader(args.datapath, limit=args.limit, split_file=args.split_file)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img), 
        batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    model = AnyNet(args)
    model = nn.DataParallel(model).cuda()

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded pretrained model '{}'".format(args.pretrained))
        else:
            print("=> no pretrained model found at '{}', Check the path then try again".format(args.pretrained))
            sys.exit(0)

    cudnn.benchmark = True
    all_outputs = evaluate(TestImgLoader, model)

    # pool = Pool(args.threads)
    # pbar = tqdm(total=(len(all_outputs)), desc="Generating Results", unit='Example')
    # def update(*a):
    #     pbar.update()
    
    # for i, image in enumerate(all_outputs):
        # sparse_and_save(args, i, image)
        # update()
        # pool.apply_async(sparse_and_save, args=(args, i, image), callback=update)

    # pool.close()
    # pool.join()
    # pbar.clear(nolock=False)
    # pbar.close()

def evaluate(dataloader, model):
    total_time = 0
    model.eval()
    for i, (imgL, imgR) in tqdm(enumerate(dataloader), desc="Generating Examples", total=(len(dataloader)), unit='Example'):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        with torch.no_grad():
            start_time = time.time()
            outputs = model(imgL, imgR)
            output3 = torch.squeeze(outputs[3], 1)
            sparse_and_save(args, i, output3.cpu())
            if i > 0:
                total_time = total_time + (time.time() - start_time)
            print("Time:    {} ms".format((time.time() - start_time)*1000))
    print("Average Time:    {} ms   ~   {} FPS".format((total_time * 1000)/(len(dataloader) - 1), (len(dataloader) - 1)/total_time))
    return


def sparse_and_save(args, i, image):
    img_cpu = cp.asarray(image)
    disp_map = img_cpu[0, :, :]
    predix = str(i).zfill(6)
    calib_file = '{}/{}.txt'.format(args.datapath +'/training/calib', predix)
    calib = Calibration(calib_file)

    """ LiDAR Generation """
    lidar = gen_lidar(disp_map, calib)

    """ Sparse LiDAR Generation """    
    sparse_point_cloud_path = args.save_path + 'sparse_point_cloud'
    if not os.path.isdir(sparse_point_cloud_path):
        os.makedirs(sparse_point_cloud_path)
    sparse_points = gen_sparse_points(lidar, H = args.H, W= args.W, slice=args.slice)
    sparse_points = sparse_points.astype(cp.float32)
    sparse_points.tofile('{}/{}.bin'.format(sparse_point_cloud_path, predix))
    return

def gen_lidar(disp_map, calib, max_high=1):
    disp_map = (disp_map*255).astype(cp.float32)/255.
    lidar = project_disp_to_points(calib, disp_map, max_high)
    lidar = cp.concatenate([lidar, cp.ones((lidar.shape[0], 1))], 1)
    lidar = lidar.astype(cp.float32)
    return lidar

def gen_sparse_points(lidar, H=64, W=512, D=700, slice=1):
    pc_velo = lidar.reshape((-1, 4))
    valid_inds =    (pc_velo[:, 0] < 120)    & \
                    (pc_velo[:, 0] >= 0)     & \
                    (pc_velo[:, 1] < 50)     & \
                    (pc_velo[:, 1] >= -50)   & \
                    (pc_velo[:, 2] < 1.5)    & \
                    (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]
    sparse_points = pto_ang_map(pc_velo, H=H, W=W, slice=slice)
    return sparse_points

def conv_disp_to_depth(disp_map, calib):
    disp_map[disp_map < 0] = 0
    baseline = 0.54
    mask = disp_map > 0
    depth = calib.f_u * baseline / (disp_map + 1. - mask)
    return depth

if __name__ == '__main__':
    main()
