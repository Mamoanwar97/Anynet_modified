import numpy as np
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

from dataloader import KITTILoader as DA
from dataloader import diy_dataset as ls
from preprocessing.kitti_util import Calibration
from preprocessing.generate_lidar import project_disp_to_points, Calibration
from preprocessing.kitti_sparsify import pto_ang_map

import argparse
from models.anynet import AnyNet
import time
import os
import sys
import tqdm

import models.anynet

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pseudoLidar/',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--max_high', type=int, default=1)

""" Kitti sparsify args """
parser.add_argument('--H', default=64, type=int)
parser.add_argument('--W', default=512, type=int)
parser.add_argument('--D', default=700, type=int)
parser.add_argument('--slice', default=1, type=int)


args = parser.parse_args()

def main():
    global args

    test_left_img, test_right_img = ls.testloader(args.datapath, limit=10)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_img, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)
    
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
            print("=> no pretrained model found at '{}'".format(args.pretrained))
            print("=> Will start from scratch.")

    args.start_epoch = 0
    cudnn.benchmark = True

    disp_images_path = args.save_path + 'disparity/'
    if not os.path.isdir(disp_images_path):
        os.makedirs(disp_images_path)

    disp_npy_path = args.save_path + 'npy/'
    if not os.path.isdir(disp_npy_path):
        os.makedirs(disp_npy_path)

    point_cloud_path = args.save_path + 'point_cloud'
    if not os.path.isdir(point_cloud_path):
        os.makedirs(point_cloud_path)
    if not os.path.isdir(sparse_point_cloud_path):
        os.makedirs(sparse_point_cloud_path)

    if args.evaluate:
        all_outputs = evaluate(TestImgLoader, model)
        for y in range(len(all_outputs)):
            """
            Array of batches loop
            """
            outputs = all_outputs[y]
            for x in range(len(outputs)):
                """
                4 stages of 1 batch

                x is stage number
                """
                output = torch.squeeze(outputs[x], 1)
                for i in range(output.size()[0]):
                    if x % 4 == 3:
                        predix = str(i*output.size()[0]+y).zfill(6)
                        img_cpu = np.asarray(output.cpu())
                        # print(np.min(img_cpu), np.max(img_cpu))

                        disp_map = img_cpu[i, :, :]
                        """ Disparity png Generation """
                        skimage.io.imsave(disp_images_path  + predix + '.png', (disp_map*255).astype('uint8'))

                        """ Disparity npy Generation """
                        np.save(disp_npy_path + predix, disp_map)

                        # if args.generate_lidar:
                        """ LiDAR Generation """
                        calib_file = '{}/{}.txt'.format(args.datapath + '/training/calib', predix)
                        calib = Calibration(calib_file)
                        disp_map = (disp_map*255).astype(np.float32)/255.
                        start_time = time.time()
                        lidar = project_disp_to_points(calib, disp_map, args.max_high)
                        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
                        lidar = lidar.astype(np.float32)
                        print("Lidar", (time.time() - start_time) * 1000, "ms")
                        lidar.tofile('{}/{}.bin'.format(point_cloud_path, predix))
                        
                        """ Sparse LiDAR Generation """
                        start_time = time.time()
                        sparse_point_cloud_path = args.save_path + 'sparse_point_cloud'
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
                        print("Sparsify", (time.time() - start_time) * 1000, "ms")
                        sparse_points.tofile('{}/{}.bin'.format(sparse_point_cloud_path, predix))
        return

def test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]

    all_outputs = []

    model.eval()
    # name = [13, 32, 36, 37, 38, 43, 46, 54, 58, 62, 75, 76, 79, 82, 92, 93, 99, 106, 108, 114, 115, 117, 124, 131, 135, 138, 139, 141, 144, 148, 159, 162, 164, 167, 176, 179, 182, 192, 193, 199]
    name = [x for x in range(4)]
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
    
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        #print('disp_l shape: {},  imgL shape: {}, imgR shape: {}'.format(disp_L.shape, imgL.shape, imgR.shape))        
        for z in range(disp_L.shape[0]):
            path = './results/output/' + str(args.datatype)
            image_name_input = path +'/d_'+ str(name[batch_idx]) + '.png'
            save_image(disp_L[z], image_name_input)

        with torch.no_grad():
            
            startTime = time.time()
            outputs, all_time = model(imgL, imgR)
            
            all_outputs.append(outputs)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())
        
        all_time = ''.join(['At Stage {}: time {:.2f} ms ~ {:.2f} FPS, error: {:.2f}%\n'.format(
            x, (all_time[x]-startTime) * 1000,  1 / ((all_time[x]-startTime)), D1s[x].val*100) for x in range(len(all_time))])
        print(all_time)
    
    #print(D1s)
    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    print('Average test 3-Pixel Error = ' + info_str)

    return all_outputs

def evaluate(dataloader, model):
    
    all_outputs = []
    model.eval()

    for imgL, imgR, _ in dataloader:
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        
        with torch.no_grad():
            # startTime = time.time()
            outputs = model(imgL, imgR)
            all_outputs.append(outputs)
            # all_time = ''.join(['At Stage {}: time {:.2f} ms ~ {:.2f} FPS\n'.format(
            #     x, (all_time[x]-startTime) * 1000,  1 / ((all_time[x]-startTime))) for x in range(len(all_time))])
            # print(all_time)

    return all_outputs

def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    errmap = torch.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
