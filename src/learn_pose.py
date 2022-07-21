# Author: Kristijan Bartol
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Sequential
import os
import numpy as np
import sys
from argparse import Namespace
from src.abstract import DSAC

sys.path.append('/general-3d-humans/')
from src.metrics import GlobalMetrics
from src.dsac import PoseDSAC
from src.dataset import init_datasets
from src.loss import MPJPELoss
from src.score import create_pose_nn
from src.const import CONNECTIVITY_DICT, PRETRAINED_PATH
from src.options import parse_args
from src.metrics import GlobalMetrics
from src.log import log_stdout
from src.visualize import store_overall_metrics, store_pose_prior_metrics, \
    store_qualitative, store_transfer_learning_metrics, store_hypotheses
   

def run(opt: Namespace, session_id: str) -> None:
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

    score_nn = create_pose_nn(
        input_size=pose_input_size, 
        hidden_layer_sizes=opt.layers_posedsac)
    
    # Set model for optimization (training).
    if not opt.cpu: score_nn = score_nn.cuda()
    opt_pose_nn = optim.Adam(
        filter(lambda p: p.requires_grad, score_nn.parameters()), lr=opt.learning_rate)
    
    # Set debugging mode.
    if opt.debug: torch.autograd.set_detect_anomaly(True)
    
    # Create "pose DSAC" (referencing "DSAC - Differentiable RANSAC for Camera Localization").
    pose_dsac = PoseDSAC(
        hyps=opt.pose_hypotheses, 
        num_joints=opt.num_joints, 
        entropy_beta=opt.entropy_beta_pose, 
        entropy_to_scores=opt.entropy_to_scores, 
        temp=opt.temp, 
        gumbel=opt.gumbel, 
        hard=opt.hard, 
        body_lengths_mode=opt.body_lengths_mode, 
        weighted_selection=opt.weighted_selection, 
        exp_beta=opt.exp_beta, 
        est_beta=opt.est_beta, 
        score_nn=score_nn, 
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
    
    if opt.test:
        _test(
            options=opt, 
            session_id=session_id,
            dsac_model=pose_dsac,
            score_nn=score_nn,
            test_dataloaders=test_dataloaders,
            global_metrics=global_metrics,
            mean=mean_3d,
            std=std_3d)
    else:
        _train(
            options=opt, 
            session_id=session_id,
            dsac_model=pose_dsac,
            score_nn=score_nn,
            optimizer=opt_pose_nn,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            global_metrics=global_metrics,
            mean=mean_3d,
            std=std_3d
        )


def _train(
        options: Namespace, 
        session_id: str, 
        dsac_model: PoseDSAC, 
        score_nn: Sequential, 
        optimizer: optim.Adam,
        train_dataloader: DataLoader, 
        valid_dataloader: DataLoader, 
        global_metrics: GlobalMetrics, 
        mean: np.ndarray, 
        std: np.ndarray
    ) -> None:
    min_mean_mpjpe = 1000.
    
    for epoch_idx in range(options.num_epochs):
        score_nn.train()

        print('############## TRAIN ################')
        
        for iteration, batch_items in enumerate(train_dataloader):
            if iteration % options.temp_step == 0 and iteration != 0:
                options.temp *= options.temp_gamma
                
            # Compute DSAC on CPU for efficiency.
            _, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]
            
            Ks = gt_Ks
            Rs = gt_Rs
            ts = gt_ts
            
            all_total_loss = 0
            num_frames = est_2d.shape[0]

            for fidx in range(num_frames):
                total_loss, global_metrics, pool_metrics = dsac_model(
                    est_2d_pose=est_2d[fidx], 
                    Ks=Ks, 
                    Rs=Rs, 
                    ts=ts, 
                    gt_3d=gt_3d[fidx], 
                    mean=mean, 
                    std=std, 
                    metrics=global_metrics)

                all_total_loss += total_loss

                if fidx % options.pose_batch_size == 0 and fidx != 0:
                    all_total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    all_total_loss = 0
                    
                log_stdout(
                    dataset='TRAIN', 
                    epoch_idx=epoch_idx, 
                    iteration=iteration, 
                    fidx=fidx, 
                    num_frames=num_frames, 
                    global_metrics=global_metrics, 
                    pool_metrics=pool_metrics, 
                    detailed=options.detailed_logs)

            if options.transfer == -1:
                store_qualitative(
                    session_id=session_id, 
                    epoch_idx=epoch_idx, 
                    iteration=iteration, 
                    dataset=options.dataset, 
                    data_type='train', 
                    pool_metrics=pool_metrics)
                store_hypotheses(
                    epoch_idx=epoch_idx, 
                    iteration=iteration, 
                    dataset=options.dataset, 
                    data_type='train', 
                    pool_metrics=pool_metrics)

            store_pose_prior_metrics(
                session_id=session_id, 
                epoch_idx=epoch_idx, 
                dataset=options.dataset, 
                data_type='train', 
                global_metrics=global_metrics)
            store_overall_metrics(
                session_id=session_id, 
                epoch_idx=epoch_idx, 
                dataset=options.dataset, 
                data_type='train', 
                global_metrics=global_metrics)

        print(f'Train epoch finished. Mean MPJPE: {global_metrics.wavg.error}')
        global_metrics.flush()
        
        print('############## VALIDATION #################')
        score_nn.eval()

        for iteration, batch_items in enumerate(valid_dataloader):
            # Load sample on CPU.
            _, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]
            
            Rs = gt_Rs
            ts = gt_ts
            
            Ks = gt_Ks
            num_frames = est_2d.shape[0]

            for fidx in range(num_frames):
                _, global_metrics, pool_metrics = dsac_model(
                    etd_2d_pose=est_2d[fidx], 
                    Ks=Ks, 
                    Rs=Rs, 
                    ts=ts, 
                    gt_3d=gt_3d[fidx], 
                    mean=mean, 
                    std=std, 
                    metrics=global_metrics)

                log_stdout(
                    dataset='VALID', 
                    epoch_idx=epoch_idx, 
                    iteration=iteration, 
                    fidx=fidx, 
                    num_frames=num_frames, 
                    global_metrics=global_metrics, 
                    pool_metrics=pool_metrics,
                    detailed=options.detailed_logs)
            
            if options.transfer == -1:
                store_qualitative(
                    session_id=session_id, 
                    epoch_idx=epoch_idx, 
                    iteration=iteration, 
                    dataset=options.dataset, 
                    dat_type='valid', 
                    pool_metrics=pool_metrics)
            
            store_overall_metrics(
                session_id=session_id, 
                epoch_idx=epoch_idx, 
                dataset=options.dataset, 
                data_type='valid', 
                global_metrics=global_metrics)
            store_pose_prior_metrics(
                session_id=session_id, 
                epoch_idx=epoch_idx, 
                dataset=options.dataset, 
                data_type='valid', 
                global_metrics=global_metrics)
            
        print(f'Validation epoch finished. Mean MPJPE: {global_metrics.wavg.error}')
        
        if global_metrics.wavg.error < min_mean_mpjpe:
            min_mean_mpjpe = global_metrics.wavg.error
            torch.save({
                'epoch': epoch_idx,
                'pose_nn_state_dict': score_nn.state_dict(),
                'opt_pose_nn_state_dict': optimizer.state_dict()
                }, 
                f'models/{session_id}_best.pt'
            )
        torch.save({
            'epoch': epoch_idx,
            'pose_nn_state_dict': score_nn.state_dict(),
            'opt_pose_nn_state_dict': optimizer.state_dict()
            }, 
            f'models/{session_id}_last.pt'
        )

        global_metrics.flush()


def _test(
        options: Namespace, 
        session_id: str,
        dsac_model: PoseDSAC,
        score_nn: Sequential,
        test_dataloaders: DataLoader,
        global_metrics: GlobalMetrics,
        mean: np.ndarray,
        std: np.ndarray
    ) -> None:
    print('########### TEST ##############')
    model_suffix = 'est' if options.use_estimated else 'known'
    score_nn_state_dict = torch.load(
        PRETRAINED_PATH.format(calib=model_suffix))['pose_nn_state_dict']
    score_nn.load_state_dict(score_nn_state_dict)
    score_nn.eval()

    mpjpe_scores_transfer = []

    for test_dataloader in test_dataloaders:
        for iteration, batch_items in enumerate(test_dataloader):
            # Load sample on CPU.
            _, est_2d, gt_3d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]
            
            Rs = gt_Rs
            ts = gt_ts
            
            Ks = gt_Ks
            num_frames = est_2d.shape[0]

            for fidx in range(num_frames):
                _, global_metrics, pool_metrics = dsac_model(
                    std_2d_pose=est_2d[fidx], 
                    Ks=Ks, 
                    Rs=Rs, 
                    ts=ts, 
                    gt_3d=gt_3d[fidx], 
                    mean=mean, 
                    std=std, 
                    metrics=global_metrics)

                log_stdout(
                    dataset='TEST', 
                    epoch_idx=0, 
                    iteration=iteration, 
                    fidx=fidx, 
                    num_frames=num_frames, 
                    global_metrics=global_metrics, 
                    pool_metrics=pool_metrics,
                    detailed=options.detailed_logs)

            if opt.transfer == -1:
                store_qualitative(
                    session_id=session_id, 
                    epoch_idx=0, 
                    iteration=iteration, 
                    dataset=opt.dataset, 
                    data_type='test', 
                    poo_metrics=pool_metrics)

            store_overall_metrics(
                session_id=session_id, 
                epoch_idx=0, 
                dataset=opt.dataset, 
                data_type='test', 
                global_metrics=global_metrics)
            store_pose_prior_metrics(
                session_id=session_id, 
                epoch_idx=0, 
                dataset=opt.dataset, 
                data_type='test', 
                global_metrics=global_metrics)
        
        print(f'Test finished. Mean MPJPE: {global_metrics.wavg.error}')

        mpjpe_scores_transfer.append(global_metrics.wavg.error)
        global_metrics.flush()

    if opt.transfer != -1:
        store_transfer_learning_metrics(
            session_id=session_id, 
            epoch_idx=0, 
            errors=mpjpe_scores_transfer)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Parse command line args.
    opt, session_id = parse_args()

    run(opt=opt, session_id=session_id)
