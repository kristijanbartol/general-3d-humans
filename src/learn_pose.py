# Author: Kristijan Bartol


from metrics import GlobalMetrics
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

from dsac import PoseDSAC
from dataset import init_datasets
from loss import MPJPELoss
from score import create_pose_nn
from mvn.utils.vis import CONNECTIVITY_DICT
from options import parse_args
from metrics import GlobalMetrics
from log import log_stdout, log_line
from visualize import store_overall_metrics, store_pose_prior_metrics, store_qualitative, \
    store_transfer_learning_metrics
    
    
if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Parse command line args.
    opt, session_id, hyperparams_string = parse_args()

    # Keep track of learning progress.
    logger = open(os.path.join('logs', f'{session_id}.txt'), 'w', 1)
    logger.write(f'{hyperparams_string}\n\n')
    logger.write('Iter\tTrain\t\tValid\t\tTest\t\tBaseline\n')

    # Create datasets.
    train_set, valid_set, test_sets = init_datasets(opt)

    mean_3d = train_set.mean_3d
    std_3d = train_set.std_3d

    # Create 3D pose loss.
    pose_loss = MPJPELoss()
    
    # Create pose scoring model.
    num_body_parts = len(CONNECTIVITY_DICT['human36m'])
    if opt.body_lengths_mode == 0:
        pose_input_size = opt.num_joints * 3
    elif opt.body_lengths_mode == 1:
        pose_input_size = opt.num_joints * 3 + num_body_parts
    elif opt.body_lengths_mode == 2:
        pose_input_size = num_body_parts

    pose_nn = create_pose_nn(input_size=pose_input_size, hidden_layer_sizes=opt.layers_posedsac)
    
    # Set model for optimization (training).
    if not opt.cpu: pose_nn = pose_nn.cuda()
    opt_pose_nn = optim.Adam(
        filter(lambda p: p.requires_grad, pose_nn.parameters()), lr=opt.learning_rate)

    # Create learning rate scheduler.
    lrs_pose_nn = optim.lr_scheduler.StepLR(opt_pose_nn, opt.lr_step, gamma=opt.lr_gamma)
    
    # Set debugging mode.
    if opt.debug: torch.autograd.set_detect_anomaly(True)
    
    # Create "pose DSAC" (referencing "DSAC - Differentiable RANSAC for Camera Localization").
    pose_dsac = PoseDSAC(
        hyps=opt.pose_hypotheses, 
        num_joints=opt.num_joints, 
        entropy_beta_pose=opt.entropy_beta_pose, 
        entropy_to_scores=opt.entropy_to_scores, 
        temp=opt.temp, 
        gumbel=opt.gumbel, 
        hard=opt.hard, 
        body_lengths_mode=opt.body_lengths_mode, 
        weighted_selection=opt.weighted_selection, 
        exp_beta=opt.exp_beta, 
        est_beta=opt.est_beta, 
        score_nn=pose_nn, 
        loss_function=pose_loss
    )
    
    # Create torch data loader.
    train_dataloader = DataLoader(train_set, shuffle=False,
                            num_workers=0, batch_size=None)
    valid_dataloader = DataLoader(valid_set, shuffle=False,
                            num_workers=0, batch_size=None)
    test_dataloaders = [DataLoader(x, shuffle=False,
                            num_workers=0, batch_size=None) for x in test_sets]

    # Initialize global metrics.
    global_metrics = GlobalMetrics(opt.dataset)
    
    # Initialize large "min" mean MPJPE for storing the best torch models.
    min_mean_mpjpe = 1000.
    
    for epoch_idx in range(opt.num_epochs):
        pose_nn.train()
        log_line = ''

        print('############## TRAIN ################')
        
        for iteration, batch_items in enumerate(train_dataloader):
            if iteration % opt.temp_step == 0 and iteration != 0:
                opt.temp *= opt.temp_gamma
                
            # Compute DSAC on CPU for efficiency.
            corresponds, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]
            
            Ks = gt_Ks
            Rs = gt_Rs
            ts = gt_ts
            
            all_total_loss = 0
            num_frames = est_2d.shape[0]

            for fidx in range(num_frames):
                total_loss, global_metrics, pool_metrics = \
                    pose_dsac(est_2d[fidx], Ks, Rs, ts, gt_3d[fidx], mean_3d, std_3d, global_metrics)

                all_total_loss += total_loss

                if fidx % opt.pose_batch_size == 0 and fidx != 0:
                    all_total_loss.backward()
                    opt_pose_nn.step()
                    opt_pose_nn.zero_grad()

                    all_total_loss = 0
                    
                log_stdout('TRAIN', epoch_idx, iteration, fidx, num_frames, global_metrics, pool_metrics)

            if opt.transfer == -1:
                store_qualitative(session_id, epoch_idx, iteration, opt.dataset, 'train', pool_metrics)

            store_pose_prior_metrics(session_id, epoch_idx, opt.dataset, 'train', global_metrics)
            store_overall_metrics(session_id, epoch_idx, opt.dataset, 'train', global_metrics)

        print(f'Train epoch finished. Mean MPJPE: {global_metrics.wavg.error}')
        log_line += f'{epoch_idx}\t\t{global_metrics.wavg.error:.4f}\t\t'
        global_metrics.flush()
        
        print('############## VALIDATION #################')
        valid_score = 0
        pose_nn.eval()

        for iteration, batch_items in enumerate(valid_dataloader):
            # Load sample on CPU.
            corresponds, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]
            
            Rs = gt_Rs
            ts = gt_ts
            
            if not opt.camdsac_only:
                Ks = gt_Ks
                num_frames = est_2d.shape[0]

                for fidx in range(num_frames):
                    _, global_metrics, pool_metrics = \
                        pose_dsac(est_2d[fidx], Ks, Rs, ts, gt_3d[fidx], mean_3d, std_3d, global_metrics)

                    log_stdout('VALID', epoch_idx, iteration, fidx, num_frames, global_metrics, pool_metrics)
                
                if opt.transfer == -1:
                    store_qualitative(session_id, epoch_idx, iteration, opt.dataset, 'valid', pool_metrics)
            
            store_overall_metrics(session_id, epoch_idx, opt.dataset, 'valid', global_metrics)
            store_pose_prior_metrics(session_id, epoch_idx, opt.dataset, 'valid', global_metrics)
            
        print(f'Validation epoch finished. Mean MPJPE: {global_metrics.wavg.error}')
        log_line += f'{global_metrics.wavg.error:.4f}\t\t'
        
        if global_metrics.wavg.error < min_mean_mpjpe:
            min_mean_mpjpe = global_metrics.wavg.error
            torch.save({
                'epoch': epoch_idx,
                'pose_nn_state_dict': pose_nn.state_dict(),
                'opt_pose_nn_state_dict': opt_pose_nn.state_dict()
                }, 
                f'models/{session_id}_best.pt'
            )
        torch.save({
            'epoch': epoch_idx,
            'pose_nn_state_dict': pose_nn.state_dict(),
            'opt_pose_nn_state_dict': opt_pose_nn.state_dict()
            }, 
            f'models/{session_id}_last.pt'
        )

        global_metrics.flush()

        if opt.test:
            print('########### TEST ##############')
            test_score = 0
            pose_nn.eval()

            mpjpe_scores_transfer = []

            counter_verification = 0
            for test_dataloader in test_dataloaders:
                for iteration, batch_items in enumerate(test_dataloader):
                    # Load sample on CPU.
                    corresponds, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]
                    
                    Rs = gt_Rs
                    ts = gt_ts
                    
                    if not opt.camdsac_only:
                        Ks = gt_Ks
                        num_frames = est_2d.shape[0]

                        for fidx in range(num_frames):
                            _, global_metrics, pool_metrics = \
                                pose_dsac(est_2d[fidx], Ks, Rs, ts, gt_3d[fidx], mean_3d, std_3d, global_metrics)

                            log_stdout('TEST', epoch_idx, iteration, fidx, num_frames, global_metrics, pool_metrics)

                            if np.isnan(global_metrics.diff_to_triang):
                                print('')

                        if opt.transfer == -1:
                            store_qualitative(session_id, epoch_idx, iteration, opt.dataset, 'test', pool_metrics)

                    store_overall_metrics(session_id, epoch_idx, opt.dataset, 'test', global_metrics)
                    store_pose_prior_metrics(session_id, epoch_idx, opt.dataset, 'test', global_metrics)
                
                print(f'Test finished. Mean MPJPE: {global_metrics.wavg.error}')
                log_line += f'{global_metrics.best.error:.4f}\t\t{(global_metrics.triang.error):.4f}\t\t' \
                    '{global_metrics.wavg.error:.4f}\t\t{(global_metrics.avg.error):.4f}\t\t{(global_metrics.random.error):.4f}'

                mpjpe_scores_transfer.append(global_metrics.wavg.error)

                logger.write(f'{log_line}\n')
                global_metrics.flush()

            if opt.transfer != -1:
                store_transfer_learning_metrics(session_id, epoch_idx, mpjpe_scores_transfer)

    logger.close()
