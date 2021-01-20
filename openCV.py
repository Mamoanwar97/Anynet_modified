from matplotlib import pyplot as plt
import cv2
import time
import numpy as np


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    errmap = np.abs(disp - gt)
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.astype(np.float32) / mask.sum().astype(np.float32)

images = [13, 32, 36, 37, 38, 43, 46, 54, 58, 62, 75, 76, 79, 82, 92, 93, 99, 106, 108, 114, 115, 117, 124, 131, 135, 138, 139, 141, 144, 148, 159, 162, 164, 167, 176, 179, 182, 192, 193, 199]

for img in images:

    imgL1 = cv2.imread('./path-to-kitti2015/training/image_2/'+ str(img).zfill(6)+'_10.png', 0)
    imgR1 = cv2.imread('./path-to-kitti2015/training/image_3/'+ str(img).zfill(6)+'_10.png', 0)
    truth = cv2.imread('./path-to-kitti2015/training/disp_occ_0/'+ str(img).zfill(6)+'_10.png', 0)

    startTime = time.time()
    
    stereo = cv2.StereoBM_create(numDisparities=192, blockSize=15)
    disparity = stereo.compute(imgL1, imgR1)

    print('At Image {}: time {:.2f} ms ~ {:.2f} FPS, error: {:.2f}%\n'.format(img, (time.time() - startTime)*1000, 1 / (time.time() - startTime), error_estimating(disparity, truth)*100))
    plt.imshow(disparity, 'gray')
    plt.show()

