import torch
import kornia

from mvn.utils.multiview import evaluate_projection, evaluate_reconstruction


class QuaternionLoss:
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


class ReprojectionLoss3D:
    '''
    Calculate loss based on the reprojection of points.
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
            points_3d, gt_Ks[0], gt_Rs[0], gt_ts[0][0], gt_ts[0][1], R_est, device=self.device)
        error_3d, _ = evaluate_reconstruction(
            points_3d, kpts_2d_projs, gt_Ks[0], gt_Rs[0], gt_ts[0][0], gt_ts[0][1], R_est)

        return error_3d


def cross_entropy_loss(est, gt):
    return -torch.log(torch.exp(est[gt.argmax()]) / torch.sum(torch.exp(est)))
