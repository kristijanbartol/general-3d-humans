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
    # Parse command line args.
    opt = parse_args()

    # Create dataset, loss, model and AutoDSAC objects.
    train_set = SparseDataset()
    loss = QuaternionLoss()
    score_nn = create_score_nn()
    auto_dsac = AutoDSAC()

    # Set ScoreNN for optimization (training).
    score_nn.train()
    opt_score_nn = optim.Adam(score_nn.parameters(), lr=opt.learningrate)
    lrs_score_nn = optim.lr_scheduler.StepLR(opt_score_nn, opt.lrstep, gamma=0.5)

    # Create torch data loader.
    dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=4, batch_size=1)

    # Train loop.
    for iteration, (point_corresponds, gt_3d, Ks, gt_Rs, gt_ts) in enumerate(dataloader):
        start_time = time.time()

        # Call AutoDSAC to obtain camera params and the ScoreNN loss expectation.
        camera_params, avg_exp_loss, best_loss = auto_dsac(
            point_corresponds, Ks, gt_3d)

        avg_exp_loss.backward()		# calculate gradients (pytorch autograd)
        score_nn.step()			    # update parameters
        score_nn.zero_grad()	    # reset gradient buffer

        end_time = time.time()-start_time

        print(f'Iteration: {iteration:.6d}, Expected Loss: {avg_exp_loss:2.2f}, '
            f'Bop Loss: {best_loss:2.2f}, Time: {end_time:.2f}s', flush=True)
