import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time

from autodsac import AutoDSAC
from dataset import SparseDataset
from loss import QuaternionLoss
from models import create_score_nn
from options import parse_args


CAM_IDXS = [3, 1]


if __name__ == '__main__':
    # Parse command line args.
    opt, sid = parse_args()

    # Keep track of training progress.
    train_log = open(os.path.join('logs', f'log_{sid}.txt'), 'w', 1)

    # Create dataset, loss and model.
    train_set = SparseDataset(opt.rootdir, CAM_IDXS, opt.num_frames, opt.num_iterations)
    loss = QuaternionLoss()
    score_nn = create_score_nn(input_size=opt.num_frames*opt.num_joints)

    # Set ScoreNN for optimization (training).
    if not opt.cpu: direct_nn = score_nn.cuda()
    score_nn.train()
    opt_score_nn = optim.Adam(
        filter(lambda p: p.requires_grad, score_nn.parameters()), lr=opt.learning_rate)
    lrs_score_nn = optim.lr_scheduler.StepLR(opt_score_nn, opt.lr_step, gamma=0.5)

    if opt.debug: torch.autograd.set_detect_anomaly(True)

    # Create AutoDSAC.
    auto_dsac = AutoDSAC(opt.hypotheses, opt.sample_size, opt.inlier_threshold, 
        opt.inlier_beta, opt.inlier_alpha, score_nn, loss)

    # Create torch data loader.
    dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=4, batch_size=1)

    # Train loop.
    for iteration, batch_items in enumerate(dataloader):
        start_time = time.time()

        point_corresponds, _, gt_Ks, gt_Rs, gt_ts = [x.cuda() for x in batch_items]

        # Call AutoDSAC to obtain camera params and the expected ScoreNN loss.
        exp_loss, camera_params, best_per_loss, best_per_score, best_per_line_dist = \
            auto_dsac(point_corresponds[0], gt_Ks, gt_Rs, gt_ts)

        exp_loss.backward()		        # calculate gradients (pytorch autograd)
        opt_score_nn.step()			    # update parameters
        opt_score_nn.zero_grad()	    # reset gradient buffer

        end_time = time.time() - start_time

        print(f'Iteration: {iteration}, Expected Loss: {exp_loss.item():.4f} \n'
            f'\tBest (per) Loss: ({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f}, {best_per_loss[2].item():.4f}) \n' 
            f'\tBest (per) Score: ({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}, {best_per_score[2].item():.4f}) \n'
            f'\tBest (per) Line Dist: ({best_per_line_dist[0].item():.4f}, {best_per_line_dist[1].item():.4f}, {best_per_line_dist[2].item():.4f})', 
            flush=True
        )
        #train_log.write(f'{iteration} {exp_loss} {top_loss}\n')

    train_log.close()
