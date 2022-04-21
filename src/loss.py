# Author: Kristijan Bartol


import torch
import kornia

from mvn.utils.multiview import evaluate_projection, evaluate_reconstruction
from metrics import rel_mpjpe
from types import LossFunction


def cross_entropy_loss(est, gt):
    return -torch.log(torch.exp(est[gt.argmax()]) / torch.sum(torch.exp(est)))


class QuaternionLoss(LossFunction):
    '''
    Calculate loss based on the difference between the rotations (in quaternions).
    '''

    def __call__(self, R_est: torch.Tensor, R_gt: torch.Tensor):
        '''Calculate the rotation loss by converting rot->quat.

        R_est -- estimated line, form: [intercept, slope]
        R_gt -- ground truth line, form: [intercept, slope]
        '''
        quat_est = kornia.rotation_matrix_to_quaternion(R_est)
        quat_gt = kornia.rotation_matrix_to_quaternion(R_gt)

        return torch.norm(quat_est - quat_gt, p=1)


class ReprojectionLoss3D(LossFunction):
    '''Calculate loss based on the reprojection of points.

    '''

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, R_est, gt_Ks, gt_Rs, gt_ts, points_3d):
        '''Calculate the rotation loss by converting rot->quat.

        R_est -- estimated line, form: [intercept, slope]
        gt_Ks -- GT intrinsics for the 2 cameras
        gt_Rs -- GT rotations for the 2 cameras
        gt_ts -- GT translations for the 2 cameras
        points_3d -- 3D points used for reprojection (could be any random set!)
        '''
        # TODO: Evaluating only rotation for now.
        kpts_2d_projs, _ = evaluate_projection(
            points_3d, gt_Ks, gt_Rs, gt_ts[0], gt_ts[1], R_est, device=self.device)
        error_3d, _ = evaluate_reconstruction(
            points_3d, kpts_2d_projs, gt_Ks, gt_Rs, gt_ts[0], gt_ts[1], R_est)

        return error_3d


class MPJPELoss(LossFunction):
    '''Calculate MPJPE loss.
    
    '''

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, est_3d, gt_3d):
        '''Calculate MPJPE loss between two 3D poses.

        est_3d -- triangulated 3D pose
        gt_3d -- GT 3D pose
        '''
        '''
        est_3d_centered = est_3d - est_3d[6, :]
        gt_3d_centered = gt_3d - gt_3d[6, :]
        return torch.mean(torch.norm(est_3d_centered - gt_3d_centered, p=2, dim=1))
        '''
        return rel_mpjpe(est_3d, gt_3d)
