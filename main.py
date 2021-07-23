import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import copy

from dsac import CameraDSAC, PoseDSAC
from dataset import SparseDataset
from loss import QuaternionLoss, ReprojectionLoss3D, MPJPELoss
from score import create_camera_nn, create_pose_nn
from options import parse_args


CAM_IDXS = [3, 1]
#CAM_IDXS = [0, 1, 2, 3]


if __name__ == '__main__':
    # Parse command line args.
    opt, sid = parse_args()

    # Keep track of training progress.
    train_log = open(os.path.join('logs', f'log_{sid}.txt'), 'w', 1)

    # Create dataset.
    train_set = SparseDataset(opt.rootdir, CAM_IDXS, opt.num_joints, opt.num_frames, opt.num_iterations)

    # Create camera and pose losses.
    camera_loss = ReprojectionLoss3D()
    pose_loss = MPJPELoss()

    # Create camera and pose scoring models.
    camera_nn = create_camera_nn(input_size=opt.num_frames * opt.num_joints, hidden_layer_sizes=opt.layers)
    pose_nn = create_pose_nn(input_size=opt.num_joints * 3)

    # Set models for optimization (training).
    if not opt.cpu: camera_nn = camera_nn.cuda()
    if not opt.cpu: pose_nn = pose_nn.cuda()

    camera_nn.train()
    pose_nn.train()

    opt_camera_nn = optim.Adam(
        filter(lambda p: p.requires_grad, camera_nn.parameters()), lr=opt.learning_rate)
    opt_pose_nn = optim.Adam(
        filter(lambda p: p.requires_grad, pose_nn.parameters()), lr=opt.learning_rate)

    lrs_camera_nn = optim.lr_scheduler.StepLR(opt_camera_nn, opt.lr_step, gamma=opt.lr_gamma)
    lrs_pose_nn = optim.lr_scheduler.StepLR(opt_pose_nn, opt.lr_step, gamma=opt.lr_gamma)

    # Set debugging mode.
    if opt.debug: torch.autograd.set_detect_anomaly(True)

    # Create DSACs.
    camera_dsac = CameraDSAC(opt.camera_hypotheses, opt.sample_size, opt.inlier_threshold, 
        opt.inlier_beta, opt.entropy_beta, opt.min_entropy, opt.entropy_to_scores, 
        opt.temp, opt.gumbel, opt.hard, camera_nn, camera_loss)
    pose_dsac = PoseDSAC(opt.pose_hypotheses, pose_nn, pose_loss)

    # Create torch data loader.
    sparse_dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=4, batch_size=None)

    # Train loop.
    for iteration, batch_items in enumerate(sparse_dataloader):
        if iteration % opt.temp_step == 0 and iteration != 0:
            opt.temp *= opt.temp_gamma

        start_time = time.time()

        # Compute DSACs on CPU for efficiency.
        corresponds, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]

        ##### CamDSAC ####################
        if not opt.posedsac_only:
            # NOTE: To test CamDSAC on a single pair, set CAM_IDXS in SparseDataset.

            # Call CamDSAC to obtain camera params and the expected ScoreNN loss.
            total_losses = []
            loss_gradients = []
            not_all_positive = False

            # Estimated camera parameters for each pair.
            Rs = []
            ts = []

            # NOTE: There are always (C-1) camera pairs.
            for pair_idx in range(1, corresponds.shape[0]):
                # Prepare pair-of-views data.
                pair_corresponds = torch.stack([corresponds[0], corresponds[pair_idx]], dim=0).transpose(0, 1)
                pair_gt_Ks = torch.stack([gt_Ks[0], gt_Ks[pair_idx]], dim=0)
                pair_gt_Rs = torch.stack([gt_Rs[0], gt_Rs[pair_idx]], dim=0)
                pair_gt_ts = torch.stack([gt_ts[0], gt_ts[pair_idx]], dim=0)

                # NOTE: GT 3D used only as any other random points for reprojection loss, for now.
                # TODO: Using GT intrinsics, for now.
                cam_dsac_result = \
                    camera_dsac(pair_corresponds, pair_gt_Ks, pair_gt_Rs, pair_gt_ts, gt_3d)
                
                # In case when all hypotheses are "Not all positive".
                if cam_dsac_result is None:
                    not_all_positive = True
                    break
                else:
                    total_loss, exp_loss, entropy_loss, est_params, best_per_loss, best_per_softmax_score, best_per_score, best_per_line_dist = cam_dsac_result

                Rs.append(est_params[0])
                ts.append(est_params[1])

                #total_loss.retain_grad()
                total_losses.append(total_loss)
                #loss_gradients.append(total_loss.backward())

            if not_all_positive:
                continue

            Rs = torch.stack(Rs, dim=0)
            ts = torch.stack(ts, dim=0)

            
            # calculate the gradients of the expected loss
            baseline = sum(total_losses) / len(total_losses) #expected loss
            '''
            for i, l in enumerate(total_losses): # substract baseline for each sample to reduce gradient variance
                loss_gradients[i] = total_losses[i].backward() * (l - baseline) / opt.samplecount
            '''

            avg_loss = torch.sum(torch.stack(total_losses, dim=0))

            #torch.autograd.backward(total_losses, loss_gradients.cuda())   # calculate gradients (pytorch autograd)
            avg_loss.backward()
            opt_camera_nn.step()			# update parameters
            opt_camera_nn.zero_grad()	    # reset gradient buffer

            end_time = time.time() - start_time

            print(f'Iteration: {iteration}, Total Loss: {total_loss.item():.4f} ({end_time:.2f}s), Expectation loss: {exp_loss:.4f}, Entropy loss: {entropy_loss:.4f}, [LR: {opt.learning_rate}, Temp: {opt.temp:.2f}]\n'
                f'\tBest (per) Loss: \t(\t{best_per_loss[0].item():.4f}, \t{best_per_loss[1].item():.4f}, \t{best_per_loss[2].item():.4f}, \t{best_per_loss[3].item():.4f}) \n' 
                f'\tBest (per) Softmax Score: (\t{best_per_softmax_score[0].item():.4f}, \t{best_per_softmax_score[1].item():.4f}, \t{best_per_softmax_score[2].item():.4f}, \t{best_per_softmax_score[3].item():.4f}) \n'
                f'\tBest (per) Score: \t(\t{best_per_score[0].item():.4f}, \t{best_per_score[1].item():.4f}, \t{best_per_score[2].item():.4f}, \t{best_per_score[3].item():.4f}) \n'
                f'\tBest (per) Line Dist: \t(\t{best_per_line_dist[0].item():.4f}, \t{best_per_line_dist[1].item():.4f}, \t{best_per_line_dist[2].item():.4f}, \t{best_per_line_dist[3].item():.4f})', 
                flush=True
            )
            ###################################
        else:
            Rs = gt_Rs
            ts = gt_ts

        # Use PoseDSAC.
        if not opt.camdsac_only:
            # TODO: Currently using known intrinsics.
            Ks = gt_Ks

            ##### PoseDSAC ####################
            start_time = time.time()
            total_pose_exp_loss = 0
            for b in range(est_2d.shape[0]):
                pose_exp_loss, pose_entropy, est_3d_pose, best_per_loss, best_per_score = pose_dsac(est_2d[b], Ks, Rs, ts, gt_3d[b])
                total_pose_exp_loss += pose_exp_loss

                print(f'Iteration: {iteration}, Expected Loss: {pose_exp_loss:.4f}, (Time: {end_time:.2f}s)\n'
                    f'\tBest (per) Loss: ({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f})\n' 
                    f'\tBest (per) Score: ({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f})',
                    flush=True
                )

            total_pose_exp_loss.backward()
            opt_pose_nn.step()
            opt_pose_nn.zero_grad()

            end_time = time.time() - start_time
            ###################################

    train_log.close()
