# Author: Kristijan Bartol
from typing import Tuple
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import Sequential
import numpy as np
import sys
from argparse import Namespace

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


def run(opt: Namespace, session_id: str) -> None:
    ''' Run function for the pose estimation model.
    
        Args:
            opt: Namespace
                The command line options namespace (see `options.py`).
            session_id: str
                Identifier used for storing the trained model.
    '''
    (pose_dsac, 
     score_nn, 
     optimizer, 
     dataloaders, 
     global_metrics, 
     mean, 
     std) = _prepare(opt)
    
    if opt.run_mode == 'eval':
        _test(
            options=opt, 
            dsac_model=pose_dsac,
            score_nn=score_nn,
            test_dataloaders=dataloaders['test'],
            global_metrics=global_metrics,
            mean=mean,
            std=std)
    else:
        _train(
            options=opt, 
            session_id=session_id,
            dsac_model=pose_dsac,
            score_nn=score_nn,
            optimizer=optimizer,
            train_dataloader=dataloaders['train'],
            valid_dataloader=dataloaders['valid'],
            global_metrics=global_metrics,
            mean=mean,
            std=std
        )   

   
def _prepare(opt: Namespace
        ) -> Tuple[
            PoseDSAC, 
            Sequential, 
            optim.Adam, 
            dict, 
            GlobalMetrics, 
            np.ndarray, 
            np.ndarray]:
    ''' Prepare model, data, optimizer, and other common vars.
    
        Args:
            opt: Namespace
                The command line options namespace (see `options.py`).
        Returns:
            dsac_model: PoseDSAC
                Pose DSAC model (see `dsac.py` -> PoseDSAC class).
            score_nn: Sequential
                The only learnable part of the model (MLP, see `score.py`).
            optimizer: optim.Adam (default)
                The standard PyTorch optimizer.
            dataloaders: dict
                Train, validation, and test dataloaders.
            global_metrics: GlobalMetrics
                The object to keep track of the global metrics (see `metrics.py`).
            mean: np.ndarray
                Mean keypoint coordinates for the dataset.
            std: np.ndarray
                Std of keypoint coordinates for the dataset.
    '''
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
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, score_nn.parameters()), lr=opt.learning_rate)
    
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
    
    dataloaders = {
        'train': train_dataloader,
        'valid': valid_dataloader,
        'test': test_dataloaders
    }

    # Initialize global metrics.
    global_metrics = GlobalMetrics(opt.dataset)
    
    return pose_dsac, score_nn, optimizer, dataloaders, global_metrics, mean_3d, std_3d


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
    ''' Run train of the pose estimation model.
    
        Args:
            options: Namespace
                The command line options namespace (see `options.py`).
            session_id: str
                Identifier used for storing the trained model.
            dsac_model: PoseDSAC
                Pose DSAC model (see `dsac.py` -> PoseDSAC class).
            score_nn: Sequential
                The only learnable part of the model (MLP, see `score.py`).
            optimizer: optim.Adam (default)
                The standard PyTorch optimizer.
            train_dataloader: dict
                Train set dataloader.
            valid_dataloader: dict
                Validation set dataloader.
            global_metrics: GlobalMetrics
                The object to keep track of the global metrics (see `metrics.py`).
            mean: np.ndarray
                Mean keypoint coordinates for the dataset.
            std: np.ndarray
                Std of keypoint coordinates for the dataset.
    '''
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

                if fidx % options.batch_size == 0 and fidx != 0:
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
        dsac_model: PoseDSAC,
        score_nn: Sequential,
        test_dataloaders: DataLoader,
        global_metrics: GlobalMetrics,
        mean: np.ndarray,
        std: np.ndarray
    ) -> None:
    ''' Run evaluation (test) of the pose estimation model.
    
        Args:
            options: Namespace
                The command line options namespace (see `options.py`).
            dsac_model: PoseDSAC
                Pose DSAC model (see `dsac.py` -> PoseDSAC class).
            score_nn: Sequential
                The only learnable part of the model (MLP, see `score.py`).
            test_dataloader: dict
                Test set dataloader.
            global_metrics: GlobalMetrics
                The object to keep track of the global metrics (see `metrics.py`).
            mean: np.ndarray
                Mean keypoint coordinates for the dataset.
            std: np.ndarray
                Std of keypoint coordinates for the dataset.
    '''
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
                    est_2d_pose=est_2d[fidx], 
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
        
        print(f'Test finished. Mean MPJPE: {global_metrics.wavg.error}')

        mpjpe_scores_transfer.append(global_metrics.wavg.error)
        global_metrics.flush()
        
        
def infer():
    pass


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Parse command line args.
    opt, session_id = parse_args()

    run(opt=opt, session_id=session_id)
