import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from .run_camera import run as run_camera
from .dsac import PoseDSAC
from .dataset import CustomTestDataset, Human36MDataset
from .score import create_pose_nn
from .const import (
    CONNECTIVITY_DICT, 
    H36M_PRETRAINED_PATH,
    INFER_DATA_DIR,
    INFER_SAVE_DIR,
    INFER_SAVE_PATH
)


def infer(opt):
    if not (os.path.exists(os.path.join(INFER_DATA_DIR, 'Rs.npy')) or \
            os.path.exists(os.path.join(INFER_DATA_DIR, 'ts.npy'))):
        Rs, ts = run_camera()
    else:
        Rs = np.load(os.path.join(INFER_DATA_DIR, 'Rs.npy'))
        ts = np.load(os.path.join(INFER_DATA_DIR, 'ts.npy'))
        
    kpts = np.load(os.path.join(INFER_DATA_DIR, 'kpts.npy'))
    
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
    
    if not opt.cpu: score_nn = score_nn.cuda()
    
    # TODO: Support custom dataset camera parameters.
    model_suffix = 'custom' if opt.custom_dataset else 'original'
    score_nn_state_dict = torch.load(
        H36M_PRETRAINED_PATH.format(calib=model_suffix))['pose_nn_state_dict']
    score_nn.load_state_dict(score_nn_state_dict)
    score_nn.eval()
    
    # Create "pose DSAC" (referencing "DSAC - Differentiable RANSAC for Camera Localization").
    dsac_model = PoseDSAC(
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
        score_nn=score_nn
    )
    
    # NOTE: Use H36M to obtain mean and std, for now.
    h36m_dataset = Human36MDataset(
        './data/human36m', 
        'test', 
        [0, 1, 2, 3], 
        custom_dataset=opt.custom_dataset, 
        num_joints=opt.num_joints, 
        num_frames=opt.num_frames, 
        num_iterations=40)
    mean = h36m_dataset.mean_3d
    std = h36m_dataset.std_3d
    custom_dataset = CustomTestDataset(kpts, Rs, ts, mean, std)
    
    infer_dataloader = DataLoader(custom_dataset, shuffle=False,
                            num_workers=0, batch_size=None)
    
    if not os.path.exists(INFER_SAVE_DIR):
        os.makedirs(INFER_SAVE_DIR)

    print('########### INFERENCE ##############')
    for idx, batch_items in enumerate(infer_dataloader):
        # Load sample on CPU.
        est_2d, gt_Ks, gt_Rs, gt_ts = [x.cpu() for x in batch_items]
        
        Rs = gt_Rs
        ts = gt_ts
        
        Ks = gt_Ks

        _, _, pool_metrics = dsac_model(
            est_2d_pose=est_2d, 
            Ks=Ks, 
            Rs=Rs, 
            ts=ts, 
            mean=mean, 
            std=std)

        np.save(
            os.path.join(INFER_SAVE_PATH.format(idx=idx)), 
            pool_metrics.wavg)
        print(f'Saved sample #{idx}...')
