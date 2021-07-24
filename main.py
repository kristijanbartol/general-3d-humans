import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import copy

from dsac import CameraDSAC, PoseDSAC
from dataset import SparseDataset, TRAIN, VALID, TEST
from loss import QuaternionLoss, ReprojectionLoss3D, MPJPELoss
from score import create_camera_nn, create_pose_nn
from mvn.utils.vis import draw_3d_pose, CONNECTIVITY_DICT
from options import parse_args


#CAM_IDXS = [3, 1]
CAM_IDXS = [0, 1, 2, 3]


if __name__ == '__main__':
    # Parse command line args.
    opt, sid = parse_args()

    # Keep track of training progress.
    train_log = open(os.path.join('logs', f'log_{sid}.txt'), 'w', 1)

    # Create datasets.
    train_set = SparseDataset(opt.rootdir, TRAIN, CAM_IDXS, opt.num_joints, opt.num_frames, opt.num_iterations)
    valid_set = SparseDataset(opt.rootdir, VALID, CAM_IDXS, opt.num_joints, opt.num_frames, opt.num_iterations)
    test_set  = SparseDataset(opt.rootdir, TEST, CAM_IDXS, opt.num_joints, opt.num_frames, opt.num_iterations)

    mean_3d = train_set.mean_3d
    std_3d = train_set.std_3d

    # Create camera and pose losses.
    camera_loss = ReprojectionLoss3D()
    pose_loss = MPJPELoss()

    # Create camera and pose scoring models.
    num_body_parts = len(CONNECTIVITY_DICT["human36m"])
    if opt.body_lengths_mode == 0:
        pose_input_size = opt.num_joints * 3
    elif opt.body_lengths_mode == 1:
        pose_input_size = opt.num_joints * 3 + num_body_parts
    elif opt.body_lengths_mode == 2:
        pose_input_size = num_body_parts

    camera_nn = create_camera_nn(input_size=opt.num_frames * opt.num_joints, hidden_layer_sizes=opt.layers_camdsac)
    pose_nn = create_pose_nn(input_size=pose_input_size, hidden_layer_sizes=opt.layers_posedsac)

    # Set models for optimization (training).
    if not opt.cpu: camera_nn = camera_nn.cuda()
    if not opt.cpu: pose_nn = pose_nn.cuda()

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
        opt.inlier_beta, opt.entropy_beta_cam, opt.min_entropy, opt.entropy_to_scores, 
        opt.temp, opt.gumbel, opt.hard, camera_nn, camera_loss)
    pose_dsac = PoseDSAC(opt.pose_hypotheses, opt.num_joints, opt.entropy_beta_pose, opt.min_entropy, opt.entropy_to_scores,
        opt.temp, opt.gumbel, opt.hard, opt.body_lengths_mode, opt.weighted_selection, pose_nn, pose_loss)

    # Create torch data loader.
    train_dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=0, batch_size=None)
    valid_dataloader = DataLoader(valid_set, shuffle=False,
                            num_workers=0, batch_size=None)
    test_dataloader  = DataLoader(test_set, shuffle=False,
                            num_workers=0, batch_size=None)

    for epoch_idx in range(opt.num_epochs):
        print('############## TRAIN ################')
        train_score = 0
        camera_nn.train()
        pose_nn.train()

        # Init PoseDSAC metrics.
        ranks = []          # hypotheses ranks
        mpjpes = []         # MPJPEs of the hypotheses
        diffs = []          # difference between the hypothesis MPJPE and best MPJPE
        top_losses = []     # losses of top hypotheses
        bottom_losses = []  # losses of worst hypotheses

        for iteration, batch_items in enumerate(train_dataloader):
            if iteration % opt.temp_step == 0 and iteration != 0:
                opt.temp *= opt.temp_gamma

            # Compute DSACs on CPU for efficiency.
            corresponds, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]

            ################## CamDSAC ####################
            if not opt.posedsac_only:
                total_losses = []
                loss_gradients = []
                exp_losses = []
                entropy_losses = []
                not_all_positive = False

                # Estimated camera parameters for each pair.
                # NOTE: To test CamDSAC on a single pair, set CAM_IDXS in SparseDataset.
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
                    # NOTE: Using GT intrinsics, for now.
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

                    if best_per_score[0] < best_per_line_dist[0]:
                        train_score += 1
                    elif best_per_score[0] > best_per_line_dist[0]:
                        train_score -= 1

                    print(f'[TRAIN] Epoch: {epoch_idx}, Iteration: {iteration}, Total Loss: {total_loss.item():.4f}, Expectation loss: {exp_loss:.4f}, Entropy loss: {entropy_loss:.4f}, [LR: {opt.learning_rate}, Temp: {opt.temp:.2f}]\n'
                        f'\tBest (per) Loss: \t(\t{best_per_loss[0].item():.4f}, \t{best_per_loss[1].item():.4f}, \t{best_per_loss[2].item():.4f}, \t{best_per_loss[3].item():.4f}) \n' 
                        f'\tBest (per) Softmax Score: (\t{best_per_softmax_score[0].item():.4f}, \t{best_per_softmax_score[1].item():.4f}, \t{best_per_softmax_score[2].item():.4f}, \t{best_per_softmax_score[3].item():.4f}) \n'
                        f'\tBest (per) Score: \t(\t{best_per_score[0].item():.4f}, \t{best_per_score[1].item():.4f}, \t{best_per_score[2].item():.4f}, \t{best_per_score[3].item():.4f}) \n'
                        f'\tBest (per) Line Dist: \t(\t{best_per_line_dist[0].item():.4f}, \t{best_per_line_dist[1].item():.4f}, \t{best_per_line_dist[2].item():.4f}, \t{best_per_line_dist[3].item():.4f})', 
                        flush=True
                    )

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

                avg_total_loss = torch.sum(torch.stack(total_losses, dim=0))

                #torch.autograd.backward(total_losses, loss_gradients.cuda())   # calculate gradients (pytorch autograd)
                avg_total_loss.backward()
                opt_camera_nn.step()			# update parameters
                opt_camera_nn.zero_grad()	    # reset gradient buffer
            else:
                Rs = gt_Rs
                ts = gt_ts
            ###############################################

            ################ PoseDSAC #####################
            if not opt.camdsac_only:
                Ks = gt_Ks

                avg_total_loss = 0

                num_frames = est_2d.shape[0]
                for fidx in range(num_frames):
                    pose_result = \
                        pose_dsac(est_2d[fidx], Ks, Rs, ts, gt_3d[fidx], mean_3d, std_3d)

                    if opt.weighted_selection:
                        total_loss, best_per_loss = pose_result
                        print(total_loss.item(), best_per_loss[0].item())
                    else:
                        total_loss, exp_loss, entropy_loss, baseline_loss, est_3d_pose, best_per_loss, best_per_score, rank, top_loss, bottom_loss = pose_result

                        # Update metrics.
                        mpjpe = best_per_score[0]
                        diff = torch.abs(best_per_score[0] - best_per_loss[0])

                        if len(ranks) == 100:
                            ranks[:-1] = ranks[1:]; ranks[-1] = rank
                            mpjpes[:-1] = mpjpes[1:]; mpjpes[-1] = mpjpe
                            diffs[:-1] = diffs[1:]; diffs[-1] = diff
                            top_losses[:-1] = top_losses[1:]; top_losses[-1] = top_loss
                            bottom_losses[:-1] = bottom_losses[1:]; bottom_losses[-1] = bottom_loss
                        else:
                            ranks.append(rank)
                            mpjpes.append(mpjpe)
                            diffs.append(diff)
                            top_losses.append(top_loss)
                            bottom_losses.append(bottom_loss)

                        mean_rank = torch.stack(ranks, dim=0).mean()
                        mean_mpjpe = torch.stack(mpjpes, dim=0).mean()
                        mean_diff = torch.stack(diffs, dim=0).mean()
                        mean_top_loss = torch.stack(top_losses, dim=0).mean()
                        mean_bottom_loss = torch.stack(bottom_losses, dim=0).mean()

                        # Log to stdout.
                        print(f'[TRAIN] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), Expectation Loss: {exp_loss:.4f}, Entropy Loss: {entropy_loss:.4f} [Rank: {mean_rank:.1f}, MPJPE: {mean_mpjpe:.2f}, Diff: {mean_diff:.2f}, Top Loss: {mean_top_loss:.2f}, Bottom Loss: {mean_bottom_loss:.2f}]\n'
                            f'\tBest (per) Loss: \t({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f})\n'
                            f'\tBest (per) Score: \t({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}) [{rank.int().item()}]\n'
                            f'\tBaseline Loss: \t\t({baseline_loss:.4f})',
                            flush=True
                        )

                    avg_total_loss += total_loss

                    if fidx % opt.pose_batch_size == 0 and fidx != 0:
                        avg_total_loss.backward()
                        opt_pose_nn.step()
                        opt_pose_nn.zero_grad()

                        avg_total_loss = 0
            ################################################
        print(f'End of epoch #{epoch_idx + 1} (validation - CamDSAC). SCORE={train_score}\n')

        print('############## VALIDATION #################')
        valid_score = 0
        camera_nn.eval()
        pose_nn.eval()

        for iteration, batch_items in enumerate(valid_dataloader):
            # Load sample.
            corresponds, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]

            # CamDSAC. #
            if not opt.posedsac_only:
                total_losses = []
                exp_losses = []
                entropy_losses = []
                not_all_positive = False

                # NOTE: There are always (C-1) camera pairs.
                for pair_idx in range(1, corresponds.shape[0]):
                    # Prepare pair-of-views data.
                    pair_corresponds = torch.stack([corresponds[0], corresponds[pair_idx]], dim=0).transpose(0, 1)
                    pair_gt_Ks = torch.stack([gt_Ks[0], gt_Ks[pair_idx]], dim=0)
                    pair_gt_Rs = torch.stack([gt_Rs[0], gt_Rs[pair_idx]], dim=0)
                    pair_gt_ts = torch.stack([gt_ts[0], gt_ts[pair_idx]], dim=0)

                    # NOTE: GT 3D used only as any other random points for reprojection loss, for now.
                    # NOTE: Using GT intrinsics, for now.
                    cam_dsac_result = \
                        camera_dsac(pair_corresponds, pair_gt_Ks, pair_gt_Rs, pair_gt_ts, gt_3d)
                    
                    # In case when all hypotheses are "Not all positive".
                    if cam_dsac_result is None:
                        not_all_positive = True
                        break
                    else:
                        total_loss, exp_loss, entropy_loss, est_params, best_per_loss, best_per_softmax_score, best_per_score, best_per_line_dist = cam_dsac_result

                    total_losses.append(total_loss)

                    if best_per_score[0] < best_per_line_dist[0]:
                        valid_score += 1
                    elif best_per_score[0] > best_per_line_dist[0]:
                        valid_score -= 1
                    
                    print(f'[VALID] Epoch: {epoch_idx}, Iteration: {iteration}, Total Loss: {total_loss.item():.4f}, Expectation loss: {exp_loss:.4f}, Entropy loss: {entropy_loss:.4f}, [LR: {opt.learning_rate}, Temp: {opt.temp:.2f}]\n'
                        f'\tBest (per) Loss: \t(\t{best_per_loss[0].item():.4f}, \t{best_per_loss[1].item():.4f}, \t{best_per_loss[2].item():.4f}, \t{best_per_loss[3].item():.4f}) \n' 
                        f'\tBest (per) Softmax Score: (\t{best_per_softmax_score[0].item():.4f}, \t{best_per_softmax_score[1].item():.4f}, \t{best_per_softmax_score[2].item():.4f}, \t{best_per_softmax_score[3].item():.4f}) \n'
                        f'\tBest (per) Score: \t(\t{best_per_score[0].item():.4f}, \t{best_per_score[1].item():.4f}, \t{best_per_score[2].item():.4f}, \t{best_per_score[3].item():.4f}) \n'
                        f'\tBest (per) Line Dist: \t(\t{best_per_line_dist[0].item():.4f}, \t{best_per_line_dist[1].item():.4f}, \t{best_per_line_dist[2].item():.4f}, \t{best_per_line_dist[3].item():.4f})', 
                        flush=True
                    )
            #############
            
            # PoseDSAC. #
            ...
            #############

        print(f'End of epoch #{epoch_idx + 1} (validation - CamDSAC). SCORE={valid_score}\n')

        ################################################


    train_log.close()
