from metrics import GlobalMetrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import kornia

from dsac import CameraDSAC, PoseDSAC
from dataset import SparseDataset, TRAIN, VALID, TEST
from loss import ReprojectionLoss3D, MPJPELoss
from score import create_camera_nn, create_pose_nn
from mvn.utils.vis import draw_3d_pose, CONNECTIVITY_DICT
from options import parse_args
from metrics import GlobalMetrics


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
    pose_dsac = PoseDSAC(opt.pose_hypotheses, opt.num_joints, opt.entropy_beta_pose, opt.min_entropy, 
        opt.entropy_to_scores, opt.temp, opt.gumbel, opt.hard, opt.body_lengths_mode, 
        opt.weighted_selection, opt.exp_beta, opt.weighted_beta, pose_nn, pose_loss)

    # Create torch data loader.
    train_dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=0, batch_size=None)
    valid_dataloader = DataLoader(valid_set, shuffle=False,
                            num_workers=0, batch_size=None)
    test_dataloader  = DataLoader(test_set, shuffle=False,
                            num_workers=0, batch_size=None)

    # Initialize global metrics.
    global_metrics = GlobalMetrics()

    for epoch_idx in range(opt.num_epochs):
        camera_score = 0
        camera_nn.train()
        pose_nn.train()

        # Init PoseDSAC metrics.
        ranks = []          # hypotheses ranks
        mpjpes = []         # MPJPEs of the hypotheses
        diffs_to_baseline = []          # difference between the hypothesis MPJPE and best MPJPE
        diffs_to_avg = []          # difference between the hypothesis MPJPE and best MPJPE
        diffs_to_random = []
        top_losses = []     # losses of top hypotheses
        bottom_losses = []  # losses of worst hypotheses

        all_mpjpes = 0.
        min_mean_mpjpe = 100.

        all_rot_errors = 0.
        all_trans_errors = 0.

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
                        total_loss, exp_loss, entropy_loss, est_params, best_per_loss, best_per_softmax_score, best_per_score, best_per_line_dist, best_line_dist_params = cam_dsac_result

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

                # NOTE: This evaluation only works for 2-camera configuration.
                R_gt_quat = kornia.rotation_matrix_to_quaternion(Rs[0])
                Rs_rel_quat = kornia.rotation_matrix_to_quaternion(gt_Rs[1])
                rot_error = torch.mean(torch.abs(R_gt_quat - Rs_rel_quat))

                all_rot_errors += rot_error.detach().numpy()
                # TODO: Estimate translation.
                all_trans_errors += 0.
                
                # variance reduction (using "baseline")
                #baseline = sum(total_losses) / len(total_losses) #expected loss
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

                all_total_loss = 0

                num_frames = est_2d.shape[0]
                for fidx in range(num_frames):
                    total_loss, global_metrics, pool_metrics = \
                        pose_dsac(est_2d[fidx], Ks, Rs, ts, gt_3d[fidx], mean_3d, std_3d, global_metrics)

                    print(f'[TRAIN] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), [MPJPE: {global_metrics.wavg.error:.2f}, Diff to [4-triang: {global_metrics.to_triang:.2f}, Average: {global_metrics.to_avg:.2f}, Random: {global_metrics.to_random:.2f}], Top Error: {global_metrics.top.error:.2f}, Bottom Error: {global_metrics.bottom.error:.2f}]\n'
                        f'\tBest (per) Loss: \t({pool_metrics.best.loss.item():.4f}, {pool_metrics.best.score.item():.4f})\n'
                        f'\tBest (per) Score: \t({pool_metrics.top.loss.item():.4f}, {pool_metrics.top.score.item():.4f})\n'
                        f'\t4-triang Loss: \t\t({pool_metrics.triang.loss.item():.4f})\n'
                        f'\tWeighted Error: \t({pool_metrics.wavg.loss.item():.4f}, {pool_metrics.avg.loss.item():.4f}, {pool_metrics.random.loss.item():.4f})',
                        flush=True
                    )

                    all_total_loss += total_loss

                    if fidx % opt.pose_batch_size == 0 and fidx != 0:
                        all_total_loss.backward()
                        opt_pose_nn.step()
                        opt_pose_nn.zero_grad()

                        all_total_loss = 0
            ################################################
        mean_rot_error = all_rot_errors / train_set.num_iterations
        mean_trans_error = all_trans_errors / train_set.num_iterations

        print(f'Train epoch finished. Mean MPJPE: {global_metrics.wavg.error}, Camera score: {camera_score} (Rot error: {mean_rot_error}, Trans error: {mean_trans_error:.2f})')

        log_line += f'{epoch_idx}\t\t{mean_rot_error:.4f}\t\t{mean_trans_error:.4f}\t\t{global_metrics.wavg.error:.4f}\t\t'

        print('############## VALIDATION #################')
        valid_score = 0
        camera_nn.eval()
        pose_nn.eval()

        all_mpjpes = 0
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
                        total_loss, exp_loss, entropy_loss, est_params, best_per_loss, best_per_softmax_score, best_per_score, best_per_line_dist, best_line_dist_params = cam_dsac_result

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
                # NOTE: This evaluation only works for 2-camera configuration.
                R2_est_quat = kornia.rotation_matrix_to_quaternion(est_params[0])
                R2_gt_quat = kornia.rotation_matrix_to_quaternion(gt_Rs[1])
                rot_error = torch.mean(torch.abs(R2_est_quat - R2_gt_quat))

                all_rot_errors += rot_error.detach().numpy()
                # TODO: Estimate translation.
                all_trans_errors += 0.
            else:
                Rs = gt_Rs
                ts = gt_ts
            #############
            
            # PoseDSAC. #
            # Init PoseDSAC metrics.
            ranks = []              # hypotheses ranks
            mpjpes = []             # MPJPEs of the hypotheses
            diffs_to_baseline = []  # difference between the hypothesis MPJPE and all-view triangulation (baseline)
            diffs_to_avg = []      # difference between the hypothesis MPJPE and best MPJPE
            diffs_to_random = []
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
                        total_loss, exp_loss, entropy_loss, baseline_loss, weighted_error, avg_error, random_error, est_3d_pose, best_per_loss, best_per_score, rank, top_loss, bottom_loss = pose_result

                        # Update metrics.
                        mpjpe = weighted_error
                        diff_to_baseline = weighted_error - baseline_loss
                        diff_to_avg = weighted_error - avg_error
                        diff_to_random = weighted_error - random_error

                        # TODO: This is only temporarily here (for test set).
                        #if mpjpe > 100.:
                        #    continue

                        if len(ranks) == 100:
                            ranks[:-1] = ranks[1:]; ranks[-1] = rank
                            mpjpes[:-1] = mpjpes[1:]; mpjpes[-1] = mpjpe
                            diffs_to_baseline[:-1] = diffs_to_baseline[1:]; diffs_to_baseline[-1] = diff_to_baseline
                            diffs_to_avg[:-1] = diffs_to_avg[1:]; diffs_to_avg[-1] = diff_to_avg
                            diffs_to_random[:-1] = diffs_to_random[1:]; diffs_to_random[-1] = diff_to_random
                            top_losses[:-1] = top_losses[1:]; top_losses[-1] = top_loss
                            bottom_losses[:-1] = bottom_losses[1:]; bottom_losses[-1] = bottom_loss
                        else:
                            ranks.append(rank)
                            mpjpes.append(mpjpe)
                            diffs_to_baseline.append(diff_to_baseline)
                            diffs_to_avg.append(diff_to_avg)
                            diffs_to_random.append(diff_to_random)
                            top_losses.append(top_loss)
                            bottom_losses.append(bottom_loss)

                        all_mpjpes += mpjpe.detach().numpy()
                        #all_mpjpes += min(weighted_error, weighted_error_top).detach().numpy()

                        mean_rank = torch.stack(ranks, dim=0).mean()
                        mean_mpjpe = torch.stack(mpjpes, dim=0).mean()
                        mean_diff_to_baseline = torch.stack(diffs_to_baseline, dim=0).mean()
                        mean_diff_to_avg = torch.stack(diffs_to_avg, dim=0).mean()
                        mean_diff_to_random = torch.stack(diffs_to_random, dim=0).mean()
                        mean_top_loss = torch.stack(top_losses, dim=0).mean()
                        mean_bottom_loss = torch.stack(bottom_losses, dim=0).mean()

                        # Log to stdout.
                        print(f'[VALIDATION] Epoch: {epoch_idx}, Iteration: {iteration} ({fidx + 1}/{num_frames} frames), [Rank: {mean_rank:.1f}, MPJPE: {mean_mpjpe:.2f}, Diff to [Baseline: {mean_diff_to_baseline:.2f}, Average: {mean_diff_to_avg:.2f}, Random: {mean_diff_to_random:.2f}], Top Loss: {mean_top_loss:.2f}, Bottom Loss: {mean_bottom_loss:.2f}]\n'
                            f'\tBest (per) Loss: \t({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f})\n'
                            f'\tBest (per) Score: \t({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}) [{rank.int().item()}]\n'
                            f'\tBaseline Loss: \t\t({baseline_loss:.4f})\n'
                            f'\tWeighted Error: \t({weighted_error:.4f}, {avg_error:.4f}, {random_error:.4f})',
                            flush=True
                        )
            #############
        mean_rot_error = all_rot_errors / valid_set.num_iterations
        mean_trans_error = all_trans_errors / valid_set.num_iterations
        if all_mpjpes > 0.:
            mean_mpjpe = all_mpjpes / (num_frames * valid_set.num_iterations)
        else:
            mean_mpjpe = 0.
        print(f'Validation finished. Mean MPJPE: {mean_mpjpe}')

        log_line += f'{mean_rot_error:.4f}\t\t{mean_trans_error:.4f}\t\t{mean_mpjpe:.4f}\t\t'

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

            all_rot_error_baselines = 0.
            all_trans_error_baselines = 0.

            all_mpjpes = 0
            all_baselines = 0
            all_best_hyp_mpjpes = 0
            all_avgs = 0
            all_randoms = 0

            counter_verification = 0
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
                            total_loss, exp_loss, entropy_loss, est_params, best_per_loss, best_per_softmax_score, best_per_score, best_per_line_dist, best_line_dist_params = cam_dsac_result

                        total_losses.append(total_loss)

                        if best_per_score[0] < best_per_line_dist[0]:
                            valid_score += 1
                        elif best_per_score[0] > best_per_line_dist[0]:
                            valid_score -= 1
                        
                        print(f'[TEST] Epoch: {epoch_idx}, Iteration: {iteration}, Total Loss: {total_loss.item():.4f}, Expectation loss: {exp_loss:.4f}, Entropy loss: {entropy_loss:.4f}, [LR: {opt.learning_rate}, Temp: {opt.temp:.2f}]\n'
                            f'\tBest (per) Loss: \t(\t{best_per_loss[0].item():.4f}, \t{best_per_loss[1].item():.4f}, \t{best_per_loss[2].item():.4f}, \t{best_per_loss[3].item():.4f}) \n' 
                            f'\tBest (per) Softmax Score: (\t{best_per_softmax_score[0].item():.4f}, \t{best_per_softmax_score[1].item():.4f}, \t{best_per_softmax_score[2].item():.4f}, \t{best_per_softmax_score[3].item():.4f}) \n'
                            f'\tBest (per) Score: \t(\t{best_per_score[0].item():.4f}, \t{best_per_score[1].item():.4f}, \t{best_per_score[2].item():.4f}, \t{best_per_score[3].item():.4f}) \n'
                            f'\tBest (per) Line Dist: \t(\t{best_per_line_dist[0].item():.4f}, \t{best_per_line_dist[1].item():.4f}, \t{best_per_line_dist[2].item():.4f}, \t{best_per_line_dist[3].item():.4f})', 
                            flush=True
                        )

                    # NOTE: This evaluation only works for 2-camera configuration.
                    R2_est_quat = kornia.rotation_matrix_to_quaternion(est_params[0])
                    R2_gt_quat = kornia.rotation_matrix_to_quaternion(gt_Rs[1])
                    rot_error = torch.mean(torch.abs(R2_est_quat - R2_gt_quat))

                    all_rot_errors += rot_error.detach().numpy()
                    # TODO: Estimate translation.
                    all_trans_errors += 0.


                    R2_est_quat_line_dist = kornia.rotation_matrix_to_quaternion(best_line_dist_params[0])
                    rot_error_line_dist = torch.mean(torch.abs(R2_est_quat_line_dist - R2_gt_quat))

                    all_rot_error_baselines += rot_error_line_dist.detach().numpy()
                    # TODO: Estimate translation.
                    all_trans_error_baselines += 0.
                else:
                    Rs = gt_Rs
                    ts = gt_ts
                #############
                
                # PoseDSAC. #
                # Init PoseDSAC metrics.
                ranks = []              # hypotheses ranks
                mpjpes = []             # MPJPEs of the hypotheses
                diffs_to_baseline = []  # difference between the hypothesis MPJPE and all-view triangulation (baseline)
                diffs_to_avg = []      # difference between the hypothesis MPJPE and best MPJPE
                diffs_to_random = []
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
                            total_loss, exp_loss, entropy_loss, baseline_loss, weighted_error, error_avg, random_error, est_3d_pose, best_per_loss, best_per_score, rank, top_loss, bottom_loss = pose_result

                            # Update metrics.
                            mpjpe = weighted_error
                            diff_to_avg = weighted_error - error_avg
                            diff_to_baseline = weighted_error - baseline_loss
                            diff_to_random = weighted_error - random_error

                            if opt.filter_bad:
                                if mpjpe > 100.:
                                    continue

                            if len(ranks) == 100:
                                ranks[:-1] = ranks[1:]; ranks[-1] = rank
                                mpjpes[:-1] = mpjpes[1:]; mpjpes[-1] = mpjpe
                                diffs_to_baseline[:-1] = diffs_to_baseline[1:]; diffs_to_baseline[-1] = diff_to_baseline
                                diffs_to_avg[:-1] = diffs_to_avg[1:]; diffs_to_avg[-1] = diff_to_avg
                                diffs_to_random[:-1] = diffs_to_random[1:]; diffs_to_random[-1] = diff_to_random
                                top_losses[:-1] = top_losses[1:]; top_losses[-1] = top_loss
                                bottom_losses[:-1] = bottom_losses[1:]; bottom_losses[-1] = bottom_loss
                            else:
                                ranks.append(rank)
                                mpjpes.append(mpjpe)
                                diffs_to_baseline.append(diff_to_baseline)
                                diffs_to_avg.append(diff_to_avg)
                                diffs_to_random.append(diff_to_random)
                                top_losses.append(top_loss)
                                bottom_losses.append(bottom_loss)

                            all_mpjpes += mpjpe.detach().numpy()
                            all_baselines += baseline_loss.detach().numpy()
                            all_best_hyp_mpjpes += best_per_loss[0].detach().numpy()
                            all_avgs += error_avg.detach().numpy()
                            all_randoms += random_error.detach().numpy()


                            counter_verification += 1

                            mean_rank = torch.stack(ranks, dim=0).mean()
                            mean_mpjpe = torch.stack(mpjpes, dim=0).mean()
                            mean_diff_to_baseline = torch.stack(diffs_to_baseline, dim=0).mean()
                            mean_diff_to_avg = torch.stack(diffs_to_avg, dim=0).mean()
                            mean_diff_to_random = torch.stack(diffs_to_random, dim=0).mean()
                            mean_top_loss = torch.stack(top_losses, dim=0).mean()
                            mean_bottom_loss = torch.stack(bottom_losses, dim=0).mean()

                            # Log to stdout.
                            print(f'[TEST] Iteration: {iteration + 1} / {len(test_set)} ({fidx + 1}/{num_frames} frames), [Rank: {mean_rank:.1f}, MPJPE: {mean_mpjpe:.2f}, Diff to [Baseline: {mean_diff_to_baseline:.2f}, Average: {mean_diff_to_avg:.2f}, Random: {mean_diff_to_random:.2f}], Top Loss: {mean_top_loss:.2f}, Bottom Loss: {mean_bottom_loss:.2f}]\n'
                                f'\tBest (per) Loss: \t({best_per_loss[0].item():.4f}, {best_per_loss[1].item():.4f})\n'
                                f'\tBest (per) Score: \t({best_per_score[0].item():.4f}, {best_per_score[1].item():.4f}) [{rank.int().item()}]\n'
                                f'\tBaseline Loss: \t\t({baseline_loss:.4f})\n'
                                f'\tWeighted Error: \t({weighted_error:.4f}, {error_avg:.4f}, {random_error:.4f})',
                                flush=True
                            )
                #############
            num_samples = test_set.preds_2d[9].shape[0] + test_set.preds_2d[11].shape[0]
            print(f'Test finished. Mean MPJPE: {(all_mpjpes / num_samples):.4f}')

            mean_rot_error = all_rot_errors / num_samples
            mean_trans_error = all_trans_errors / num_samples

            mean_rot_error_baseline = all_rot_error_baselines / num_samples

            if all_mpjpes > 0.:
                mean_mpjpe = all_mpjpes / num_samples
                mean_baseline = all_baselines / num_samples
                mean_best_hyp_mpjpe = all_best_hyp_mpjpes / num_samples
                mean_avg = all_avgs / num_samples
                mean_random = all_randoms / num_samples
            else:
                mean_mpjpe = 0.
                mean_baseline = 0.
                mean_best_hyp_mpjpe = 0.
                mean_avg = 0.
                mean_random = 0.

            log_line += f'{mean_rot_error:.4f}\t\t{mean_trans_error:.4f}\t\t{mean_rot_error_baseline:.4f}\t\t{mean_best_hyp_mpjpe.item():.4f}\t\t{(mean_baseline):.4f}\t\t{mean_mpjpe:.4f}\t\t{(mean_avg):.4f}\t\t{(mean_random):.4f}'
            logger.write(f'{log_line}\n')
            ################################################

    logger.close()
