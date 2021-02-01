import argparse
import os
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
from preprocessing.generate_lidar import project_disp_to_points, project_depth_to_points, Calibration
from dataloader import KITTILoader as DA
from dataloader import diy_dataset as ls


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

args = parser.parse_args()

def main():
    global args

    test_left_img, test_right_img, test_left_disp = ls.testloader(args.datapath)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
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
    if args.evaluate:
        print(len(TestImgLoader))
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
                    """
                    images in 1 batch
                    i is the image index in the batch
                    """
                    if x % 4 == 3:
                        path = args.save_path
                        predix = str(i*output.size()[0]+y).zfill(6)

                        img_cpu = np.asarray(output.cpu())
                        # print(np.min(img_cpu), np.max(img_cpu))

                        disp_map = img_cpu[i, :, :]
                        # disp_map = np.clip(img_cpu[i, :, :], 0, 2**16)
                        # disp_map = (img_cpu[i, :, :] - np.min(img_cpu[i, :, :])) / (np.max(img_cpu[i, :, :]) - np.min(img_cpu[i, :, :]))
                        # disp_map = (disp_map*255).astype(np.uint8)
                        # assert os.path.isdir(path + 'disparity/')
                        skimage.io.imsave(path + 'disparity/' + predix + '.png',(disp_map*256).astype('uint16'))
                        # ssc.imsave(path + 'disparity/' + predix + '.png', disp_map)

                        print(np.min(disp_map), np.max(disp_map))
                        np.save(path + 'npy/' + predix, disp_map)
                        calib_file = '{}/{}.txt'.format(args.datapath + '/training/calib', predix)
                        calib = Calibration(calib_file)

                        disp_map = (disp_map*255).astype(np.uint16)/255.
                        # print(disp_map.dtype)
                        lidar = project_disp_to_points(calib, disp_map, args.max_high)

                        lidar = np.concatenate([lidar, np.ones((lidar.shape[0], 1))], 1)
                        lidar = lidar.astype(np.float32)

                        lidar.tofile('{}/{}.bin'.format(path + 'point_cloud', predix))
                        print('Finish Depth {}'.format(predix))
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
