import argparse
import os
import sys
import numpy as np
import skimage
import skimage.io
import scipy.misc as ssc
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torchvision.utils import save_image
import time
import utils.logger as logger
import torch.backends.cudnn as cudnn
from preprocessing.generate_lidar import project_disp_to_points, Calibration
from preprocessing.kitti_sparsify import pto_ang_map
from dataloader import KITTILoader as DA
from dataloader import diy_dataset as ls
import tqdm

import models.anynet

parser = argparse.ArgumentParser(description='Evaluating Anynet')

parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--pretrained', type=str, default=None, help='pretrained model path')
parser.add_argument('--save_path', type=str, default='results/pseudoLidar/', help='the path of saving checkpoints and log')

""" LiDAR args """
parser.add_argument('--max_high', type=int, default=1)

""" OPTIONS """
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--generate_lidar', action='store_true', help='with generate lidar or not')

""" Anynet modal args """
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')

""" Kitti sparsify args """
parser.add_argument('--H', default=64, type=int)
parser.add_argument('--W', default=512, type=int)
parser.add_argument('--D', default=700, type=int)
parser.add_argument('--slice', default=1, type=int)

args = parser.parse_args()

def main():
    global args

    test_left_img, test_right_img = ls.testloader(args.datapath)

    TestImgLoader = torch.utils.data.DataLoader(DA.myImageFloder(test_left_img, test_right_img, test_left_img, False, evaluating=True), batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    model = models.anynet.AnyNet(args)
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

    for i, single_batch in tqdm.tqdm(enumerate(all_outputs), ascii=True, desc="Generating Results", total=(len(all_outputs)), unit='Example'):
        for j, stage in enumerate(single_batch):
            output = torch.squeeze(stage, 1)
            img_cpu = np.asarray(output.cpu())
            disp_map = img_cpu[0, :, :]
            if j % 4 == 3:
                predix = str(i).zfill(6)

                """ Disparity png Generation """
                disp_images_path = args.save_path + 'disparity/'
                if not os.path.isdir(disp_images_path):
                    os.makedirs(disp_images_path)
                skimage.io.imsave(disp_images_path  + predix + '.png', (disp_map*256).astype('uint16'))

                """ Disparity npy Generation """
                disp_npy_path = args.save_path + 'npy/'
                if not os.path.isdir(disp_npy_path):
                    os.makedirs(disp_npy_path)
                np.save(disp_npy_path + predix, disp_map)

                if args.generate_lidar:
                    """ LiDAR Generation """
                    point_cloud_path = args.save_path + 'point_cloud'
                    if not os.path.isdir(point_cloud_path):
                        os.makedirs(point_cloud_path)
                    calib_file = '{}/{}.txt'.format(args.datapath + '/training/calib', predix)
                    calib = Calibration(calib_file)
                    lidar = project_disp_to_points(calib, disp_map, args.max_high)
                    lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
                    lidar = lidar.astype(np.float32)
                    lidar.tofile('{}/{}.bin'.format(point_cloud_path, predix))
                    
                    """ Sparse LiDAR Generation """
                    sparse_point_cloud_path = args.save_path + 'sparse_point_cloud'
                    if not os.path.isdir(sparse_point_cloud_path):
                        os.makedirs(sparse_point_cloud_path)
                    pc_velo = lidar.reshape((-1, 4))
                    valid_inds =    (pc_velo[:, 0] < 120)    & \
                                    (pc_velo[:, 0] >= 0)     & \
                                    (pc_velo[:, 1] < 50)     & \
                                    (pc_velo[:, 1] >= -50)   & \
                                    (pc_velo[:, 2] < 1.5)    & \
                                    (pc_velo[:, 2] >= -2.5)
                    pc_velo = pc_velo[valid_inds]
                    sparse_points = pto_ang_map(pc_velo, H=args.H, W=args.W, slice=args.slice)
                    sparse_points = sparse_points.astype(np.float32)
                    sparse_points.tofile('{}/{}.bin'.format(sparse_point_cloud_path, predix))
    return

def evaluate(dataloader, model):
    all_outputs = []
    model.eval()
    times = 0

    for i, (imgL, imgR) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        with torch.no_grad():
            startTime = time.time()
            outputs = model(imgL, imgR)
            all_outputs.append(outputs)
            if i > 0:
                times = times + (time.time()-startTime)
    
    Average_time = (times * 1000)/(len(all_outputs) - 1) #millisecond
    Average_FPS =   (len(all_outputs) - 1) / times
    print('\nAverage Time: {:.2f} ms ~ {:.2f} FPS\n'.format(Average_time, Average_FPS))
    return all_outputs

def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

if __name__ == '__main__':
    main()
