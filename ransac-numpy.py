import numpy as np
import os
from scipy.spatial.transform import Rotation as R

from mvn.utils.multiview import find_rotation_matrices_numpy, \
    solve_four_solutions_numpy, distance_between_projections_numpy, \
    evaluate_projection_numpy, evaluate_reconstruction_numpy


SUBJECT_IDX = 9
IDXS = [2, 1]

DATA_ROOT = f'./results/S{SUBJECT_IDX}/'

IMGS_PATH = os.path.join(DATA_ROOT, 'all_images.npy')
PRED_PATH = os.path.join(DATA_ROOT, 'all_2d_preds.npy')
GT_PATH = os.path.join(DATA_ROOT, 'all_3d_gt.npy')
KS_BBOX_PATH = os.path.join(DATA_ROOT, 'Ks_bboxed.npy')
K_PATH = os.path.join(DATA_ROOT, 'Ks.npy')
R_PATH = os.path.join(DATA_ROOT, 'Rs.npy')
T_PATH = os.path.join(DATA_ROOT, 'ts.npy')
BBOX_PATH = os.path.join(DATA_ROOT, 'all_bboxes.npy')


M = 50             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 2000           # trials
eps = 0.75           # outlier probability
S = 100              # sample size
#I = (1 - eps) * P  # number of inliers condition
I = 0
D = 1.             # distance criterion
T = int(N / 20)              # number of top candidates to use


def load_camera_params(subject_idx, cam_idxs):
    SUBJECT_IDXS = {
        1:  0,
        5:  1,
        6:  2,
        7:  3,
        8:  4,
        9:  5,
        11: 6
    }
    labels = np.load(
        '/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', 
        allow_pickle=True).item()
    camera_params = labels['cameras'][SUBJECT_IDXS[subject_idx]]

    Ks, Rs, ts = [], [], []
    for cam_idx in cam_idxs:
        Ks.append(camera_params[cam_idx][2])
        Rs.append(camera_params[cam_idx][0])
        ts.append(camera_params[cam_idx][1])

    Ks = np.stack(Ks, axis=0)
    Rs = np.stack(Rs, axis=0)
    ts = np.stack(ts, axis=0)

    return Ks, Rs, ts


if __name__ == '__main__':
    # Load predictions, ground truth and camera parameters.
    all_2d_preds = np.load(PRED_PATH)
    all_3d_gt = np.load(GT_PATH)
    bboxes = np.load(BBOX_PATH)
    Ks, Rs, ts = load_camera_params(SUBJECT_IDX, IDXS)

    # NOTE: Currently using all available frames for sampling in each run.
    frame_selection = np.arange(all_2d_preds.shape[0])

    all_2d_preds = all_2d_preds[frame_selection][:, IDXS]
    all_3d_gt = all_3d_gt[frame_selection]
    bboxes = bboxes[frame_selection][:, IDXS]

    # Unbbox keypoints.
    bbox_height = np.abs(bboxes[:, :, 0, 0] - bboxes[:, :, 1, 0])
    all_2d_preds *= np.expand_dims(
        np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
    all_2d_preds += np.expand_dims(bboxes[:, :, 0, :], axis=2)

    # All points stacked along a single dimension.
    point_corresponds = np.concatenate(
        np.split(all_2d_preds, all_2d_preds.shape[0], axis=0), 
        axis=2)[0].swapaxes(0, 1)


    ########### Obtain R_rel_gt, t_rel_gt and scale ############
    R_rel_gt = Rs[1] @ np.linalg.inv(Rs[0])
    (kpts1_gt, kpts2_gt), _ = evaluate_projection_numpy(all_3d_gt, Ks, Rs, ts, R_rel_gt, ts[1])
    kpts1_gt = kpts1_gt.reshape((-1, 2))
    kpts2_gt = kpts2_gt.reshape((-1, 2))

    t_rel_gt = -Rs[1] @ np.linalg.inv(Rs[0]) @ ts[0] + ts[1]
    
    dists = distance_between_projections_numpy(kpts1_gt, kpts2_gt, Ks, Rs, R_rel_gt, ts[0], ts[1])
    condition = dists < D
    num_inliers = (condition).sum()

    print(f'Mean distances between corresponding lines (GT all): {dists.mean()}')

    assert(num_inliers == point_corresponds.shape[0])

    inliers = np.stack((kpts1_gt, kpts2_gt), axis=1)
    try:
        _, _, t_rel, _ = find_rotation_matrices_numpy(inliers, Ks)
    except Exception as ex:
        print(f'[GT data + GT camera params] {ex}')

    scale = (t_rel_gt / t_rel).mean()
    ############################################################

    counter = 0
    # RANSAC loop.
    for i in range(N):
        # Selecting indexes (sampling).
        selected_idxs = np.random.choice(
            np.arange(point_corresponds.shape[0]), size=S)

        # Find camera parameters using 8-point algorithm.
        # TODO: Rename: find_rotation_matrices* -> find_camera_parameters*.
        R_est1, R_est2, t_rel_est, _ = find_rotation_matrices_numpy(
            point_corresponds[selected_idxs], Ks)

        try:
        # Select correct rotation and find translation sign.
            t_rel_est *= scale
            R_est, t_rel_est = solve_four_solutions_numpy(
                point_corresponds, Ks, Rs, ts, 
                (R_est1, R_est2), t_rel_est)
        except:
            print('Not all positive')
            continue

        # Find inliers based on 3D line distances.
        line_dists = distance_between_projections_numpy(
            point_corresponds[:, 0], point_corresponds[:, 1], 
            Ks, Rs, R_est, ts[0], t_rel_est)

        condition = line_dists < D
        num_inliers = (condition).sum()

        # Evaluate projection in 2D.
        kpts_2d_projs, error_2d = evaluate_projection_numpy(
            all_3d_gt, Ks, Rs, ts, R_est, t_rel_est)
        # Evaluate reconstruction in 3D.
        error_3d, _ = evaluate_reconstruction_numpy(
            all_3d_gt, kpts_2d_projs, Ks, Rs, ts, R_est, t_rel_est)
        # Evaluate rotation (compare quaternions).
        R_est_quat = R.from_dcm(R_est).as_quat()
        R_rel_gt_quat = R.from_dcm(R_rel_gt).as_quat()
        quat_norm = np.linalg.norm(R_rel_gt_quat - R_est_quat, ord=1)

        print(f'{counter}. ({num_inliers}, {line_dists.mean():.3f}) -> '
            f'{quat_norm:.2f} {error_2d:.2f}, {error_3d:.2f}')

        counter += 1