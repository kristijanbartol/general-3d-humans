from enum import auto
import torch
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn
from torch.utils.data import DataLoader

import time

from skimage.io import imsave

from autodsac import AutoDSAC
from dataset import SparseDataset
from loss import QuaternionLoss, ReprojectionLoss3D
from models import create_score_nn
from options import parse_args


if __name__ == '__main__':
    opt = parse_args()

    train_set = SparseDataset()
    loss = QuaternionLoss()
    score_nn = create_score_nn()
    auto_dsac = AutoDSAC()

    dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=4, batch_size=1)

    # Train loop.
    for iteration, (point_corresponds, gt_3d, Ks, gt_Rs, gt_ts) in enumerate(dataloader):
        start_time = time.time()
        camera_params, avg_exp_loss, best_loss = auto_dsac(
            point_corresponds, Ks, gt_3d)

        avg_exp_loss.backward()		# calculate gradients (pytorch autograd)
        score_nn.step()			    # update parameters
        score_nn.zero_grad()	    # reset gradient buffer

        end_time = time.time()-start_time

        print(f'Iteration: {iteration:.6d}, Expected Loss: {avg_exp_loss:2.2f}, '
            f'Bop Loss: {best_loss:2.2f}, Time: {end_time:.2f}s', flush=True)
