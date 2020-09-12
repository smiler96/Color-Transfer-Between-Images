import cv2
import numpy as np
import argparse
import os

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='source/s5.bmp')
    parser.add_argument('--target', type=str, default='target/t5.bmp')
    parser.add_argument('--clip', type=bool, default=True)

    parser.add_argument('--result', type=str, default=None)
    hparams = parser.parse_args()
    return hparams

def cal_mean_std(x):
    '''

    :param x: l,a,b channels
    :return: mean, std
    '''
    mean_ = np.zeros(3)
    std_ = np.zeros(3)
    for i in range(3):
        mean_[i] = np.mean(x[:, :, i])
        std_[i] = np.std(x[:, :, i])
    return mean_, std_

def transfer_color(src, target, clip=True, init=False):
    '''
    :param src: bgr img
    :param target:  bgr img
    :param clip: bool
    :return: transfer
    '''
    src = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    src = src.astype(np.float32)
    target = target.astype(np.float32)

    sMean, sStd = cal_mean_std(src)
    tMean, tStd = cal_mean_std(target)

    transfer = np.zeros(src.shape)
    for i in range(0, 3):
        if init:
            s = sStd[i] / tStd[i]
        else:
            s = tStd[i] / sStd[i]
        transfer[:, :, i] = (src[:, :, i] - sMean[i]) * s + tMean[i]
    if clip:
        transfer = np.clip(transfer, 0, 255)

    transfer = np.uint8(transfer)
    transfer = cv2.cvtColor(transfer, cv2.COLOR_LAB2BGR)
    return transfer

if __name__ == '__main__':
    hparams = get_params()
    src = cv2.imread(hparams.src)
    target = cv2.imread(hparams.target)

    res = transfer_color(src, target, clip=hparams.clip)

    if hparams.result == None:
        sname = os.path.basename(hparams.src).split('.')[0]
        tname = os.path.basename(hparams.target).split('.')[0]
        name = sname+ '_' + tname + '.bmp'
        hparams.result = os.path.join('transfer', name)
    # res = np.concatenate([src, res, target], axis=1)
    cv2.imwrite(hparams.result, res)