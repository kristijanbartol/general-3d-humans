import torch
import torch.optim as optim
import torchvision.utils as vutils

import os
import time
import numpy
import warnings
import argparse

from skimage.io import imsave

from autodsac import AutoDSAC
from dataset import SparseDataset
from loss import ReprojectionLoss3D
from models import ScoreCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script creates a toy problem of '
        'fitting line parameters (slope+intercept) to synthetic images showing line segments, '
        'noise and distracting circles. Two networks are trained in parallel and compared: '
        'DirectNN predicts the line parameters directly (two output neurons). PointNN predicts '
        'a number of 2D points to which the line parameters are subsequently fitted using '
        'differentiable RANSAC (DSAC). The script will produce a sequence of images that '
        'illustrate the training process for both networks.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--receptivefield', '-rf', type=int, default=65, choices=[65, 51, 37, 29, 15, 0],
	    help='receptive field size of the PointNN, i.e. one point prediction is made for each image patch of this size, different receptive fields are achieved by different striding strategies, 0 means global, i.e. the full image, the DirectNN will always use 0 (global)')

    parser.add_argument('--capacity', '-c', type=int, default=4, 
        help='controls the model capactiy of both networks (PointNN and DirectNN), it is a multiplicative factor for the number of channels in each network layer')

    parser.add_argument('--hypotheses', '-hyps', type=int, default=64, 
        help='number of line hypotheses sampled for each image')

    parser.add_argument('--inlierthreshold', '-it', type=float, default=0.05, 
        help='threshold used in the soft inlier count. Its measured in relative image size (1 = image width)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=0.5, 
        help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

    parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0, 
        help='scaling factor within the sigmoid of the soft inlier count')

    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, 
        help='learning rate')

    parser.add_argument('--lrstep', '-lrs', type=int, default=2500, 
        help='cut learning rate in half each x iterations')

    parser.add_argument('--lrstepoffset', '-lro', type=int, default=30000, 
        help='keep initial learning rate for at least x iterations')

    parser.add_argument('--batchsize', '-bs', type=int, default=32, 
        help='training batch size')

    parser.add_argument('--trainiterations', '-ti', type=int, default=50000, 
        help='number of training iterations (= parameter updates)')

    parser.add_argument('--imagesize', '-is', type=int, default=64, 
        help='size of input images generated, images are square')

    parser.add_argument('--storeinterval', '-si', type=int, default=1000, 
        help='store network weights and a prediction vizualisation every x training iterations')

    parser.add_argument('--valsize', '-vs', type=int, default=9, 
        help='number of validation images used to vizualize predictions')

    parser.add_argument('--valthresh', '-vt', type=float, default=5, 
        help='threshold on the line loss for vizualizing correctness of predictions')

    parser.add_argument('--cpu', '-cpu', action='store_true',
        help='execute networks on CPU. Note that (RANSAC) line fitting anyway runs on CPU')

    parser.add_argument('--session', '-sid', default='',
        help='custom session name appended to output files. Useful to separate different runs of the program')

    opt = parser.parse_args()

    if len(opt.session) > 0: opt.session = '_' + opt.session
    sid = 'rf%d_c%d_h%d_t%.2f%s' % (opt.receptivefield, opt.capacity, opt.hypotheses, opt.inlierthreshold, opt.session)

    dataset = SparseDataset()
    loss = ReprojectionLoss3D()
    score_cnn = ScoreCNN()
    auto_dsac = AutoDSAC()
