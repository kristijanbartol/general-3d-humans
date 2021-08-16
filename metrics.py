import torch


def mpjpe(est, gt):
    return torch.mean(torch.norm(est - gt, p=2, dim=1))


def center_pelvis(pose_3d):
    return pose_3d - pose_3d[6, :]


def rel_mpjpe(est, gt):
    est_centered = center_pelvis(est)
    gt_centered = center_pelvis(gt)

    return mpjpe(est_centered, gt_centered)
