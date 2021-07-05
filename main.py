import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import copy

from dsac import CameraDSAC, PoseDSAC
from dataset import SparseDataset
from loss import QuaternionLoss, ReprojectionLoss3D
from score import create_camera_nn, create_pose_nn
from options import parse_args


#CAM_IDXS = [3, 1]
CAM_IDXS = [0, 1, 2, 3]


if __name__ == '__main__':
    # Parse command line args.
    opt, sid = parse_args()

    # Keep track of training progress.
    train_log = open(os.path.join('logs', f'log_{sid}.txt'), 'w', 1)

    # Create dataset.
    train_set = SparseDataset(opt.rootdir, CAM_IDXS, opt.num_frames, opt.num_iterations)

    # Create camera and pose losses.
    camera_loss = ReprojectionLoss3D()
    pose_loss = nn.MSELoss()

    # Create camera and pose scoring models.
    camera_nn = create_camera_nn(input_size=opt.num_frames*opt.num_joints)
    pose_nn = create_pose_nn(input_size=opt.num_joints)

    # Set models for optimization (training).
    if not opt.cpu: camera_nn = camera_nn.cuda()
    if not opt.cpu: pose_nn = pose_nn.cuda()

    camera_nn.train()
    pose_nn.train()

    opt_camera_nn = optim.Adam(
        filter(lambda p: p.requires_grad, camera_nn.parameters()), lr=opt.learning_rate)
    opt_pose_nn = optim.Adam(
        filter(lambda p: p.requires_grad, pose_nn.parameters()), lr=opt.learning_rate)

    lrs_camera_nn = optim.lr_scheduler.StepLR(opt_camera_nn, opt.lr_step, gamma=0.5)
    lrs_pose_nn = optim.lr_scheduler.StepLR(opt_pose_nn, opt.lr_step, gamma=0.5)

    # Set debugging mode.
    if opt.debug: torch.autograd.set_detect_anomaly(True)

    # Create DSACs.
    camera_dsac = CameraDSAC(opt.camera_hypotheses, opt.sample_size, opt.inlier_threshold, 
        opt.inlier_beta, opt.inlier_alpha, camera_nn, camera_loss)
    pose_dsac = PoseDSAC(opt.pose_hypotheses, pose_nn, pose_loss)

    # Create torch data loader.
    sparse_dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=4, batch_size=None)

    # Train loop.
    for iteration, batch_items in enumerate(sparse_dataloader):
        start_time = time.time()

        if not opt.use_gt_cameras:
            ##### AutoDSAC ####################
            # NOTE: To test AutoDSAC on a single pair, set CAM_IDXS in SparseDataset.
            # Compute AutoDSAC on CPU for efficiency.
            point_corresponds, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]

            # Call AutoDSAC to obtain camera params and the expected ScoreNN loss.
            # NOTE: There are always (C-1) camera pairs.
            Rs = [gt_Rs[0][0]]
            ts = [gt_ts[0][0]]

            total_cam_exp_loss = 0

            for pair_idx in range(point_corresponds.shape[0]):
                # NOTE: GT 3D used only as any other random points for reprojection loss, for now.
                # TODO: Using GT intrinsics, for now.
                cam_exp_loss, entropy, est_params, best_per_loss, best_per_score, best_per_line_dist = \
                    camera_dsac(point_corresponds[pair_idx], 
                                gt_Ks[pair_idx], 
                                gt_Rs[pair_idx], 
                                gt_ts[pair_idx], 
                                gt_3d
                )
                Rs.append(est_params[0])
                ts.append(est_params[1])

                total_cam_exp_loss += cam_exp_loss

            total_cam_exp_loss.backward()		        # calculate gradients (pytorch autograd)
            opt_camera_nn.step()			# update parameters
            opt_camera_nn.zero_grad()	    # reset gradient buffer
            ###################################
        else:
            Rs = gt_Rs
            ts = gt_ts

        # TODO: Currently using known intrinsics.
        # Select init camera intrinsics and then (C-1) other intrinsics.
        Ks = torch.cat([gt_Ks[0, 0]] + [gt_Ks[x, 1] for x in range(gt_Ks.shape[0])], dim=0)

        ##### PoseDSAC ####################
        for b in range(est_2d.shape[0]):
            pose_exp_loss, est_3d_pose = pose_dsac(est_2d, Ks, Rs, ts, gt_3d)

        ###################################

        end_time = time.time() - start_time

        print(f'Iteration: {iteration}, Expected Loss: {total_cam_exp_loss.item():.4f} ({end_time:.2f}s), Entropy: {entropy:.4f}\n'
            f'\tBest (per) Loss: ({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f}, {best_per_loss[2].item():.4f}) \n' 
            f'\tBest (per) Score: ({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}, {best_per_score[2].item():.4f}) \n'
            f'\tBest (per) Line Dist: ({best_per_line_dist[0].item():.4f}, {best_per_line_dist[1].item():.4f}, {best_per_line_dist[2].item():.4f})', 
            flush=True
        )
        #train_log.write(f'{iteration} {exp_loss} {top_loss}\n')

    train_log.close()
