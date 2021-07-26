import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from dsac import CameraDSAC, PoseDSAC
from dataset import SparseDataset, TRAIN, VALID, TEST
from loss import ReprojectionLoss3D, MPJPELoss
from score import create_camera_nn, create_pose_nn
from mvn.utils.vis import draw_3d_pose, CONNECTIVITY_DICT
from options import parse_args


#CAM_IDXS = [3, 1]
CAM_IDXS = [0, 1, 2, 3]


if __name__ == '__main__':
    # Parse command line args.
    opt, session_id, hyperparams_string = parse_args()

    # Keep track of learning progress.
    logger = open(os.path.join('logs', f'{session_id}.txt'), 'w', 1)
    logger.write(f'{hyperparams_string}\n\n')
    logger.write('Iter\tTrain\t\tValid\t\tTest\t\tBaseline\n')

    # Create datasets.
    train_set = SparseDataset(opt.rootdir, TRAIN, CAM_IDXS, opt.num_joints, opt.num_frames, opt.train_iterations)
    valid_set = SparseDataset(opt.rootdir, VALID, CAM_IDXS, opt.num_joints, opt.num_frames, opt.valid_iterations)
    test_set  = SparseDataset(opt.rootdir, TEST, CAM_IDXS, opt.num_joints, opt.num_frames, None)

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
        camera_score = 0
        camera_nn.train()
        pose_nn.train()

        # Init PoseDSAC metrics.
        ranks = []          # hypotheses ranks
        mpjpes = []         # MPJPEs of the hypotheses
        diffs_to_best = []          # difference between the hypothesis MPJPE and best MPJPE
        diffs_to_baseline = []          # difference between the hypothesis MPJPE and best MPJPE
        top_losses = []     # losses of top hypotheses
        bottom_losses = []  # losses of worst hypotheses

        all_mean_mpjpe = 0.
        min_mean_mpjpe = 100.
        log_line = ''

        print('############## TRAIN ################')

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

                    total_losses.append(total_loss)

                    if best_per_score[0] < best_per_line_dist[0]:
                        camera_score += 1
                    elif best_per_score[0] > best_per_line_dist[0]:
                        camera_score -= 1

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
                        total_loss, exp_loss, entropy_loss, baseline_loss, weighted_error, weighted_error_top, est_3d_pose, best_per_loss, best_per_score, rank, top_loss, bottom_loss = pose_result

                        # Update metrics.
                        mpjpe = weighted_error
                        diff_to_best = weighted_error - best_per_loss[0]
                        diff_to_baseline = weighted_error - baseline_loss

                        if len(ranks) == 100:
                            ranks[:-1] = ranks[1:]; ranks[-1] = rank
                            mpjpes[:-1] = mpjpes[1:]; mpjpes[-1] = mpjpe
                            diffs_to_best[:-1] = diffs_to_best[1:]; diffs_to_best[-1] = diff_to_best
                            diffs_to_baseline[:-1] = diffs_to_baseline[1:]; diffs_to_baseline[-1] = diff_to_baseline
                            top_losses[:-1] = top_losses[1:]; top_losses[-1] = top_loss
                            bottom_losses[:-1] = bottom_losses[1:]; bottom_losses[-1] = bottom_loss
                        else:
                            ranks.append(rank)
                            mpjpes.append(mpjpe)
                            diffs_to_best.append(diff_to_best)
                            diffs_to_baseline.append(diff_to_baseline)
                            top_losses.append(top_loss)
                            bottom_losses.append(bottom_loss)

                        all_mean_mpjpe += mpjpe

                        mean_rank = torch.stack(ranks, dim=0).mean()
                        mean_mpjpe = torch.stack(mpjpes, dim=0).mean()
                        mean_diff_to_best = torch.stack(diffs_to_best, dim=0).mean()
                        mean_diff_to_baseline = torch.stack(diffs_to_baseline, dim=0).mean() # .............
                        mean_top_loss = torch.stack(top_losses, dim=0).mean()
                        mean_bottom_loss = torch.stack(bottom_losses, dim=0).mean()

                        # Log to stdout.
                        print(f'[TRAIN] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), [Rank: {mean_rank:.1f}, MPJPE: {mean_mpjpe:.2f}, Diff to baseline: {mean_diff_to_baseline:.2f}, Diff to best: {mean_diff_to_best:.2f}, Top Loss: {mean_top_loss:.2f}, Bottom Loss: {mean_bottom_loss:.2f}]\n'
                            f'\tBest (per) Loss: \t({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f})\n'
                            f'\tBest (per) Score: \t({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}) [{rank.int().item()}]\n'
                            f'\tBaseline Loss: \t\t({baseline_loss:.4f})\n'
                            f'\tWeighted Error: \t({weighted_error:.4f}, {weighted_error_top:.4f})',
                            flush=True
                        )

                    avg_total_loss += total_loss

                    if fidx % opt.pose_batch_size == 0 and fidx != 0:
                        avg_total_loss.backward()
                        opt_pose_nn.step()
                        opt_pose_nn.zero_grad()

                        avg_total_loss = 0
            ################################################
        mean_mpjpe = all_mean_mpjpe / (num_frames * train_set.num_iterations)
        print(f'Train epoch finished. Mean MPJPE: {mean_mpjpe}, Camera score: {camera_score}')

        log_line += f'{epoch_idx}\t\t{mean_mpjpe:.4f}\t\t'

        print('############## VALIDATION #################')
        valid_score = 0
        camera_nn.eval()
        pose_nn.eval()

        all_mean_mpjpe = 0
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
            else:
                Rs = gt_Rs
                ts = gt_ts
            #############
            
            # PoseDSAC. #
            # Init PoseDSAC metrics.
            ranks = []              # hypotheses ranks
            mpjpes = []             # MPJPEs of the hypotheses
            diffs_to_best = []      # difference between the hypothesis MPJPE and best MPJPE
            diffs_to_baseline = []  # difference between the hypothesis MPJPE and all-view triangulation (baseline)
            top_losses = []         # losses of top hypotheses
            bottom_losses = []      # losses of worst hypotheses

            if not opt.camdsac_only:
                Ks = gt_Ks
                num_frames = est_2d.shape[0]

                for fidx in range(num_frames):
                    pose_result = \
                        pose_dsac(est_2d[fidx], Ks, Rs, ts, gt_3d[fidx], mean_3d, std_3d)

                    if opt.weighted_selection:
                        total_loss, best_per_loss = pose_result
                        print(total_loss.item(), best_per_loss[0].item())
                    else:
                        total_loss, exp_loss, entropy_loss, baseline_loss, weighted_error, weighted_error_top, est_3d_pose, best_per_loss, best_per_score, rank, top_loss, bottom_loss = pose_result

                        # Update metrics.
                        mpjpe = weighted_error
                        diff_to_best = weighted_error - best_per_loss[0]
                        diff_to_baseline = weighted_error - baseline_loss

                        # TODO: This is only temporarily here (for test set).
                        #if mpjpe > 100.:
                        #    continue

                        if len(ranks) == 100:
                            ranks[:-1] = ranks[1:]; ranks[-1] = rank
                            mpjpes[:-1] = mpjpes[1:]; mpjpes[-1] = mpjpe
                            diffs_to_best[:-1] = diffs_to_best[1:]; diffs_to_best[-1] = diff_to_best
                            diffs_to_baseline[:-1] = diffs_to_baseline[1:]; diffs_to_baseline[-1] = diff_to_baseline
                            top_losses[:-1] = top_losses[1:]; top_losses[-1] = top_loss
                            bottom_losses[:-1] = bottom_losses[1:]; bottom_losses[-1] = bottom_loss
                        else:
                            ranks.append(rank)
                            mpjpes.append(mpjpe)
                            diffs_to_best.append(diff_to_best)
                            diffs_to_baseline.append(diff_to_baseline)
                            top_losses.append(top_loss)
                            bottom_losses.append(bottom_loss)

                        all_mean_mpjpe += mpjpe

                        mean_rank = torch.stack(ranks, dim=0).mean()
                        mean_mpjpe = torch.stack(mpjpes, dim=0).mean()
                        mean_diff_to_best = torch.stack(diffs_to_best, dim=0).mean()
                        mean_diff_to_baseline = torch.stack(diffs_to_baseline, dim=0).mean()
                        mean_top_loss = torch.stack(top_losses, dim=0).mean()
                        mean_bottom_loss = torch.stack(bottom_losses, dim=0).mean()

                        # Log to stdout.
                        print(f'[VALIDATION] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), [Rank: {mean_rank:.1f}, MPJPE: {mean_mpjpe:.2f}, Diff to baseline: {mean_diff_to_baseline:.2f}, Diff to best: {mean_diff_to_best:.2f}, Top Loss: {mean_top_loss:.2f}, Bottom Loss: {mean_bottom_loss:.2f}]\n'
                            f'\tBest (per) Loss: \t({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f})\n'
                            f'\tBest (per) Score: \t({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}) [{rank.int().item()}]\n'
                            f'\tBaseline Loss: \t\t({baseline_loss:.4f})\n'
                            f'\tWeighted Error: \t({weighted_error:.4f}, {weighted_error_top:.4f})',
                            flush=True
                        )
            #############

        mean_mpjpe = all_mean_mpjpe / (num_frames * valid_set.num_iterations)
        print(f'Validation finished. Mean MPJPE: {mean_mpjpe}')

        log_line += f'{mean_mpjpe:.4f}\t\t'

        if mean_mpjpe < min_mean_mpjpe:
            min_mean_mpjpe = mean_mpjpe
            torch.save({
                'epoch': epoch_idx,
                'camera_nn_state_dict': camera_nn.state_dict(),
                'opt_camera_nn_state_dict': opt_camera_nn.state_dict(),
                'pose_nn_state_dict': pose_nn.state_dict(),
                'opt_pose_nn_state_dict': opt_pose_nn.state_dict()
                }, 
                f'models/{session_id}_best.pt'
            )
        torch.save({
            'epoch': epoch_idx,
            'camera_nn_state_dict': camera_nn.state_dict(),
            'opt_camera_nn_state_dict': opt_camera_nn.state_dict(),
            'pose_nn_state_dict': pose_nn.state_dict(),
            'opt_pose_nn_state_dict': opt_pose_nn.state_dict()
            }, 
            f'models/{session_id}_last.pt'
        )
        ################################################

        if opt.test:
            print('########### TEST ##############')
            test_score = 0
            camera_nn.eval()
            pose_nn.eval()

            all_mpjpes = 0
            all_baselines = 0
            for iteration, batch_items in enumerate(test_dataloader):
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
                else:
                    Rs = gt_Rs
                    ts = gt_ts
                #############
                
                # PoseDSAC. #
                # Init PoseDSAC metrics.
                ranks = []              # hypotheses ranks
                mpjpes = []             # MPJPEs of the hypotheses
                diffs_to_best = []      # difference between the hypothesis MPJPE and best MPJPE
                diffs_to_baseline = []  # difference between the hypothesis MPJPE and all-view triangulation (baseline)
                top_losses = []         # losses of top hypotheses
                bottom_losses = []      # losses of worst hypotheses

                if not opt.camdsac_only:
                    Ks = gt_Ks
                    num_frames = est_2d.shape[0]

                    for fidx in range(num_frames):
                        pose_result = \
                            pose_dsac(est_2d[fidx], Ks, Rs, ts, gt_3d[fidx], mean_3d, std_3d)

                        if opt.weighted_selection:
                            total_loss, best_per_loss = pose_result
                            print(total_loss.item(), best_per_loss[0].item())
                        else:
                            total_loss, exp_loss, entropy_loss, baseline_loss, weighted_error, weighted_error_top, est_3d_pose, best_per_loss, best_per_score, rank, top_loss, bottom_loss = pose_result

                            # Update metrics.
                            mpjpe = weighted_error
                            diff_to_best = weighted_error - best_per_loss[0]
                            diff_to_baseline = weighted_error - baseline_loss


                            if opt.filter_bad:
                                if mpjpe > 100.:
                                    continue

                            if len(ranks) == 100:
                                ranks[:-1] = ranks[1:]; ranks[-1] = rank
                                mpjpes[:-1] = mpjpes[1:]; mpjpes[-1] = mpjpe
                                diffs_to_best[:-1] = diffs_to_best[1:]; diffs_to_best[-1] = diff_to_best
                                diffs_to_baseline[:-1] = diffs_to_baseline[1:]; diffs_to_baseline[-1] = diff_to_baseline
                                top_losses[:-1] = top_losses[1:]; top_losses[-1] = top_loss
                                bottom_losses[:-1] = bottom_losses[1:]; bottom_losses[-1] = bottom_loss
                            else:
                                ranks.append(rank)
                                mpjpes.append(mpjpe)
                                diffs_to_best.append(diff_to_best)
                                diffs_to_baseline.append(diff_to_baseline)
                                top_losses.append(top_loss)
                                bottom_losses.append(bottom_loss)

                            all_mpjpes += mpjpe
                            all_baselines += baseline_loss

                            mean_rank = torch.stack(ranks, dim=0).mean()
                            mean_mpjpe = torch.stack(mpjpes, dim=0).mean()
                            mean_diff_to_best = torch.stack(diffs_to_best, dim=0).mean()
                            mean_diff_to_baseline = torch.stack(diffs_to_baseline, dim=0).mean()
                            mean_top_loss = torch.stack(top_losses, dim=0).mean()
                            mean_bottom_loss = torch.stack(bottom_losses, dim=0).mean()

                            # Log to stdout.
                            print(f'[TEST] Iteration: {iteration + 1} / {len(test_set)} ({fidx + 1}/{num_frames} frames), [Rank: {mean_rank:.1f}, MPJPE: {mean_mpjpe:.2f}, Diff to baseline: {mean_diff_to_baseline:.2f}, Diff to best: {mean_diff_to_best:.2f}, Top Loss: {mean_top_loss:.2f}, Bottom Loss: {mean_bottom_loss:.2f}]\n'
                                f'\tBest (per) Loss: \t({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f})\n'
                                f'\tBest (per) Score: \t({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}) [{rank.int().item()}]\n'
                                f'\tBaseline Loss: \t\t({baseline_loss:.4f})\n'
                                f'\tWeighted Error: \t({weighted_error:.4f}, {weighted_error_top:.4f})',
                                flush=True
                            )
                #############
            num_samples = test_set.preds_2d[9].shape[0] + test_set.preds_2d[11].shape[0]
            print(f'Test finished. Mean MPJPE: {(all_mean_mpjpe / num_samples):.4f}')

            log_line += f'{all_mpjpes / num_samples:.4f}\t\t{(all_baselines / num_samples):.4f}'
            logger.write(f'{log_line}\n')
            ################################################

    logger.close()
