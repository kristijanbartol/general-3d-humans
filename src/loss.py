# Author: Kristijan Bartol


import torch
import kornia
import sys

sys.path.append('/general-3d-humans/')
from src.multiview import evaluate_projection, evaluate_reconstruction
from src.metrics import rel_mpjpe
from src.abstract import LossFunction


def cross_entropy_loss(est, gt):
    return -torch.log(torch.exp(est[gt.argmax()]) / torch.sum(torch.exp(est)))


class QuaternionLoss(LossFunction):
    '''Calculate loss based on the difference between the rotations (in quaternions).
    
        The difference between the quaternions is simply a 1-norm 
        (absolute difference) between the values.
    '''

    def __call__(self, 
                 R_est: torch.Tensor, 
                 R_gt: torch.Tensor
        ) -> torch.Tensor:
        '''Calculate the rotation loss by converting rot->quat.

            Parameters
            ----------
            :param R_est: estimated line, form: [intercept, slope]
            :param R_gt: ground truth line, form: [intercept, slope]
            :return: absolute (1-norm) difference between the rotation quaternions
        '''
        quat_est = kornia.rotation_matrix_to_quaternion(R_est)
        quat_gt = kornia.rotation_matrix_to_quaternion(R_gt)

        return torch.norm(quat_est - quat_gt, p=1)


class ReprojectionLoss3D(LossFunction):
    '''Calculate loss based on the reprojection of points.

    '''

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, 
                 R_est: torch.Tensor, 
                 gt_Ks: torch.Tensor, 
                 gt_Rs: torch.Tensor, 
                 gt_ts: torch.Tensor, 
                 points_3d
        ) -> torch.Tensor:
        '''Calculate the rotation loss by converting rot->quat.

            Parameters
            ----------
            :param R_est: estimated line, form: [intercept, slope]
            :param gt_Ks: GT intrinsics for the 2 cameras
            :param gt_Rs: GT rotations for the 2 cameras
            :param gt_ts: GT translations for the 2 cameras
            :param points_3d: 3D points used for reprojection (could be any random set!)
            :return: 3D error (scalar)
        '''
        # TODO: Evaluating only rotation for now.
        kpts_2d_projs, _ = evaluate_projection(
            points_3d, gt_Ks, gt_Rs, gt_ts[0], gt_ts[1], R_est, device=self.device)
        error_3d, _ = evaluate_reconstruction(
            points_3d, kpts_2d_projs, gt_Ks, gt_Rs, gt_ts[0], gt_ts[1], R_est)

        return error_3d


class MPJPELoss(LossFunction):
    '''Calculate MPJPE (mean per-joint position error) loss.
    
        The metric is defined in "Human3.6M: Large Scale Datasets 
        and Predictive Methods for 3D Human Sensing in Natural Environments".
    '''

    def __init__(self, 
                 device: str = 'cpu'
        ) -> None:
        '''MPJPE loss constructor.
        
        '''
        self.device = device

    def __call__(self, 
                 est_3d: torch.Tensor, 
                 gt_3d: torch.Tensor
        ) -> torch.Tensor:
        '''Calculate MPJPE loss between two 3D poses.

        :param est_3d: estimated 3D pose (bsize, J, 3)
        :param gt_3d: ground-truth 3D pose (bsize, J, 3)
        :return: relative MPJPE (bsize,)
        '''
        return rel_mpjpe(est_3d, gt_3d, device=self.device)
