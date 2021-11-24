
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

from mvn.utils.multiview import find_rotation_matrices_numpy, \
    solve_four_solutions_numpy, distance_between_projections_numpy, \
    evaluate_projection_numpy, evaluate_reconstruction_numpy


SUBJECT_IDX = 9
IDXS = [2, 1]

DATA_ROOT = f'./results/human36m/S{SUBJECT_IDX}/'

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
N = 200           # trials
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


def fundamental(M, IDXS):
    # Load predictions, ground truth and camera parameters.
    all_2d_preds = np.load(PRED_PATH)
    all_3d_gt = np.load(GT_PATH)
    bboxes = np.load(BBOX_PATH)
    Ks, Rs, ts = load_camera_params(SUBJECT_IDX, IDXS)

    # NOTE: Currently using all available frames for sampling in each run.
    #frame_selection = np.arange(all_2d_preds.shape[0])
    frame_selection = np.random.choice(
            np.arange(all_2d_preds.shape[0]), size=M)

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

    #print(f'Mean distances between corresponding lines (GT all): {dists.mean()}')

    assert(num_inliers == point_corresponds.shape[0])

    inliers = np.stack((kpts1_gt, kpts2_gt), axis=1)
    try:
        _, _, t_rel, _ = find_rotation_matrices_numpy(inliers, Ks)
    except Exception as ex:
        print(f'[GT data + GT camera params] {ex}')

    scale = (t_rel_gt / t_rel).mean()
    ############################################################


    metrics = []
    counter = 0

    # RANSAC loop.
    for i in range(N):
        # Selecting indexes (sampling).
        selected_idxs = np.random.choice(
            np.arange(point_corresponds.shape[0]), size=S, replace=False)

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
            #print('Not all positive')
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
        error_R = np.linalg.norm(R_rel_gt_quat - R_est_quat, ord=1)

        R_est_abs = R_est @ Rs[0]

        # Evaluate translation.
        t2_est = -Rs[1] @ np.linalg.inv(Rs[0]) @ ts[0] + t_rel_est
        error_t = np.linalg.norm(t2_est - t_rel_gt, ord=2)

        if error_3d > 1000.:
            continue
        metrics.append([line_dists.mean(), error_R, error_t, error_2d, error_3d, R_est, t_rel_est])

        #print(f'{counter}. ({num_inliers}, {line_dists.mean():.3f}) -> '
        #    f'{error_R:.4f} {error_2d:.2f}, {error_3d:.2f}')

        counter += 1

    metrics = np.array(metrics)
    our_metrics = metrics[np.argmin(metrics[:, 0])]
    print(f'Our score: {our_metrics[4]:.2f}')


    # Using all indexes.
    selected_idxs = np.arange(point_corresponds.shape[0])

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
        #print('Not all positive')
        pass

    # Evaluate projection in 2D.
    kpts_2d_projs, error_2d = evaluate_projection_numpy(
        all_3d_gt, Ks, Rs, ts, R_est, t_rel_est)

    # Evaluate reconstruction in 3D.
    error_3d, _ = evaluate_reconstruction_numpy(
        all_3d_gt, kpts_2d_projs, Ks, Rs, ts, R_est, t_rel_est)

    # Evaluate rotation (compare quaternions).
    R_est_quat = R.from_dcm(R_est).as_quat()
    R_rel_gt_quat = R.from_dcm(R_rel_gt).as_quat()
    error_R = np.linalg.norm(R_rel_gt_quat - R_est_quat, ord=1)

    # Evaluate translation.
    t2_est = -Rs[1] @ np.linalg.inv(Rs[0]) @ ts[0] + t_rel_est
    error_t = np.linalg.norm(t2_est - t_rel_gt, ord=2)

    if error_3d > 1000.:
        return None, None

    vanilla_metrics = [error_R, error_t, error_2d, error_3d]

    print(f'Vanilla score: {vanilla_metrics[3]:.2f}')

    return our_metrics[1:], vanilla_metrics
    

def compare_to_vanilla():
    all_our_means = []
    all_our_stds = []
    all_vanilla_means = []
    all_vanilla_stds = []

    num_frames_range = range(10, 100)

    for M in num_frames_range:
        our_repeated = []
        vanilla_repeated = []
        for j in range(10):
            our_metrics, vanilla_metrics = fundamental(M=M, IDXS=IDXS)
            if vanilla_metrics is None:
                continue
            our_repeated.append(our_metrics)
            vanilla_repeated.append(vanilla_metrics)
        all_our_means.append(np.array(our_repeated).mean(axis=0))
        all_our_stds.append(np.array(our_repeated).std(axis=0))
        
        all_vanilla_means.append(np.array(vanilla_repeated).mean(axis=0))
        all_vanilla_stds.append(np.array(vanilla_repeated).std(axis=0))

    all_our_means = np.array(all_our_means)
    all_our_stds = np.array(all_our_stds)

    all_vanilla_means = np.array(all_vanilla_means)
    all_vanilla_stds = np.array(all_vanilla_stds)


    x = np.array(num_frames_range)
    our_y = all_our_means[:, 3]
    our_upper = our_y + all_our_stds[:, 3]
    our_lower = np.clip(our_y - all_our_stds[:, 3], a_min=0, a_max=None)

    vanilla_y = all_vanilla_means[:, 3]
    vanilla_upper = np.clip(vanilla_y + all_vanilla_stds[:, 3], a_min=None, a_max=80.)
    vanilla_lower = np.clip(vanilla_y - all_vanilla_stds[:, 3], a_min=0, a_max=None)


    our_plt, = plt.plot(x, our_y, color='dodgerblue', label='Our model')
    plt.fill_between(x, our_upper, our_lower, color='crimson', alpha=0.2)

    vanilla_plt, = plt.plot(x, vanilla_y, color='darkorange', label='Vanilla 8-point')
    plt.fill_between(x, vanilla_upper, vanilla_lower, color='lightgreen', alpha=0.2)

    plt.legend(handles=[our_plt, vanilla_plt])

    plt.xlabel('Number of frames')
    plt.ylabel('3D error')

    plt.style.use('seaborn')

    plt.savefig(f'./results/8-point_{IDXS[0]}_{IDXS[1]}_new.png')
    np.save(f'./results/our_scores_{IDXS[0]}_{IDXS[1]}.npy', all_our_means)
    np.save(f'./results/vanilla_scores_{IDXS[0]}_{IDXS[1]}.npy', all_vanilla_means)


def estimate_params():
    np.random.seed(2022)
    est_Rs = [ np.eye(3) ]
    est_ts = [ np.zeros((3, 1)) ]
    for cam_idxs in [(0, 1), (0, 2), (0, 3)]:
        our_metrics, _ = fundamental(M=M, IDXS=cam_idxs)
        R_est, t_rel_est = our_metrics[-2:]
        print(f'Selected params error ({cam_idxs}): {our_metrics[-3]}mm')
        est_Rs.append(R_est)
        est_ts.append(t_rel_est)
    np.save('./results/est_Rs.npy', np.array(est_Rs, dtype=np.float32))
    np.save('./results/est_ts.npy', np.array(est_ts, dtype=np.float32))


if __name__ == '__main__':
    #compare_to_vanilla()
    estimate_params()


'''

import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from mvn.utils.multiview import find_rotation_matrices_numpy, \
    solve_four_solutions_numpy, distance_between_projections_numpy, \
    evaluate_projection_numpy, evaluate_reconstruction_numpy


SUBJECT_IDX = 1
IDXS = [3, 1]

DATA_ROOT = f'./results/human36m/S{SUBJECT_IDX}/'

IMGS_PATH = os.path.join(DATA_ROOT, 'all_images.npy')
PRED_PATH = os.path.join(DATA_ROOT, 'all_2d_preds0.npy')
GT_PATH = os.path.join(DATA_ROOT, 'all_3d_gt0.npy')
KS_BBOX_PATH = os.path.join(DATA_ROOT, 'Ks_bboxed0.npy')
K_PATH = os.path.join(DATA_ROOT, 'Ks.npy')
R_PATH = os.path.join(DATA_ROOT, 'Rs.npy')
T_PATH = os.path.join(DATA_ROOT, 'ts.npy')
BBOX_PATH = os.path.join(DATA_ROOT, 'all_bboxes0.npy')


M = 60             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 200           # trials
eps = 0.75           # outlier probability
S = 50              # sample size
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
    all_3d_gt_orig = np.load(GT_PATH)
    bboxes = np.load(BBOX_PATH)
    Ks, Rs, ts = load_camera_params(SUBJECT_IDX, IDXS)

    our_scores = []
    vanilla_scores = []

    all_2d_preds_orig = all_2d_preds[:, IDXS]
    bboxes_orig = bboxes[:, IDXS]

    for M in range(30, 60):

        ##### Our approach #######
        # NOTE: Currently using all available frames for sampling in each run.
        #frame_selection = np.arange(all_2d_preds.shape[0])
        frame_selection = np.random.choice(
                np.arange(all_2d_preds.shape[0]), size=M)

        all_2d_preds = all_2d_preds_orig[frame_selection]
        all_3d_gt = all_3d_gt_orig[frame_selection]
        bboxes = bboxes_orig[frame_selection]

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

        #print(f'Mean distances between corresponding lines (GT all): {dists.mean()}')

        assert(num_inliers == point_corresponds.shape[0])

        inliers = np.stack((kpts1_gt, kpts2_gt), axis=1)
        try:
            _, _, t_rel, _ = find_rotation_matrices_numpy(inliers, Ks)
        except Exception as ex:
            print(f'[GT data + GT camera params] {ex}')

        scale = (t_rel_gt / t_rel).mean()
        ############################################################


        counter = 0
        num_invalid = 0
        metrics = []

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
                #print('Not all positive')
                num_invalid += 1
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

            mean_line_dists = line_dists.mean()
            if error_3d > 1000.:
                continue
            #print(f'{counter}. ({num_inliers}, {line_dists.mean():.3f}) -> '
            #    f'{quat_norm:.4f} {error_2d:.2f}, {error_3d:.2f}')

            metrics.append([mean_line_dists, error_3d])
            counter += 1

        if num_invalid == N:
            continue

        metrics = np.array(metrics)
        our_score = metrics[np.argmin(metrics[:, 0])]
        print(f'Our score: {our_score[1]} (num_invalid={num_invalid})')
        
        ##########################



        #### Vanilla 8-point. ####

        # Selecting indexes (sampling).
        selected_idxs = np.random.choice(
            np.arange(point_corresponds.shape[0]), size=M*J)

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
            #print('Not all positive')
            continue

        # Find inliers based on 3D line distances.
        line_dists = distance_between_projections_numpy(
            point_corresponds[:, 0], point_corresponds[:, 1], 
            Ks, Rs, R_est, ts[0], t_rel_est)

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

        if error_3d > 1000.:
            continue

        print(f'Vanilla score: {error_3d}')

        our_scores.append(our_score)
        vanilla_scores.append(error_3d)

    our_scores = np.array(our_scores)
    vanilla_scores = np.array(vanilla_scores)

    print(f'Our scores: {our_scores.mean()}, {our_scores.std()}')
    print(f'Vanilla scores: {vanilla_scores.mean()}, {vanilla_scores.std()}')

    #plt.plot(list(num_frames_range), our_scores)
    #plt.plot(list(num_frames_range), vanilla_scores)
    #plt.style.use('seaborn')

    #plt.savefig('8-point.png')

    print('')
'''