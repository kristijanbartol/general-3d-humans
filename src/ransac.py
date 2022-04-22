# Author: Kristijan Bartol


import numpy as np
import torch
import kornia
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
import copy

from multiview import find_rotation_matrices, compare_rotations, \
    evaluate_projection, evaluate_reconstruction, distance_between_projections, solve_four_solutions, \
    get_positive_points_mask


SUBJECT_IDX = 9
IDXS = [1, 0]


DATA_ROOT = f'./results/S{SUBJECT_IDX}/'

# TODO: Update paths and concatenate numpy arrays.
IMGS_PATH = os.path.join(DATA_ROOT, 'all_images0.npy')
PRED_PATH = os.path.join(DATA_ROOT, 'all_2d_preds0.npy')
GT_PATH = os.path.join(DATA_ROOT, 'all_3d_gt0.npy')
KS_BBOX_PATH = os.path.join(DATA_ROOT, 'Ks_bboxed0.npy')
K_PATH = os.path.join(DATA_ROOT, 'Ks.npy')
R_PATH = os.path.join(DATA_ROOT, 'Rs.npy')
T_PATH = os.path.join(DATA_ROOT, 'ts.npy')
BBOX_PATH = os.path.join(DATA_ROOT, 'all_bboxes0.npy')

M = 50             # number of frames
J = 17              # number of joints
P = M * J           # total number of point correspondences    
N = 200           # trials
eps = 0.75           # outlier probability
S = 100              # sample size
#I = (1 - eps) * P  # number of inliers condition
I = 0
D = .5             # distance criterion
T = int(N/20)              # number of top candidates to use



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
    labels = np.load('/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', allow_pickle=True).item()
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
    with torch.no_grad():
        Ks, Rs, ts = load_camera_params(SUBJECT_IDX, IDXS)
        
        all_2d_preds = np.load(PRED_PATH)
        all_3d_gt = torch.tensor(np.load(GT_PATH), device='cuda', dtype=torch.float32)
        #Ks = torch.unsqueeze(torch.tensor(np.load(K_PATH), device='cuda', dtype=torch.float32), dim=0)[:, IDXS]
        Ks = torch.unsqueeze(torch.tensor(Ks, device='cuda', dtype=torch.float32), dim=0)
        #Rs = torch.unsqueeze(torch.tensor(np.load(R_PATH), device='cuda', dtype=torch.float32), dim=0)[:, IDXS]
        Rs = torch.unsqueeze(torch.tensor(Rs, device='cuda', dtype=torch.float32), dim=0)
        #ts = torch.unsqueeze(torch.tensor(np.load(T_PATH), device='cuda', dtype=torch.float32), dim=0)[:, IDXS]
        ts = torch.unsqueeze(torch.tensor(ts, device='cuda', dtype=torch.float32), dim=0)
        bboxes = np.load(BBOX_PATH)

        frame_selection = np.random.choice(np.arange(all_2d_preds.shape[0]), size=M)
        #frame_selection = np.arange(50)
        #frame_selection = np.arange(all_2d_preds.shape[0])  # using all available frames for sampling in each run

        all_2d_preds = all_2d_preds[frame_selection][:, IDXS]
        all_3d_gt = all_3d_gt[frame_selection]
        bboxes = bboxes[frame_selection][:, IDXS]

        # Unbbox keypoints.
        bbox_height = np.abs(bboxes[:, :, 0, 0] - bboxes[:, :, 1, 0])
        all_2d_preds *= np.expand_dims(np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
        all_2d_preds += np.expand_dims(bboxes[:, :, 0, :], axis=2)

        # All points stacked along a single dimension.
        point_corresponds = torch.tensor(np.concatenate(np.split(all_2d_preds, all_2d_preds.shape[0], axis=0), axis=2)[0], 
            device='cuda', dtype=torch.float32).transpose(0, 1)

        all_2d_preds = torch.tensor(all_2d_preds, device='cuda', dtype=torch.float32)

        #K_step_x = 0.1 * Ks[0][0][0][2]
        #K_step_y = 0.1 * Ks[0][0][1][2]

        real_fx = copy.deepcopy(Ks[0][0][0][0])
        real_fy = copy.deepcopy(Ks[0][0][1][1])

        #Ks[0][0][0][0] = Ks[0][0][0][2]
        #Ks[0][0][1][1] = Ks[0][0][1][2]

        invalid_counter = 0
        intrinsics_found = False
        for K_i in range(1):
            if intrinsics_found:
                break

            print(f'Current intrinsics: ({Ks[0][0][0][0]:.2f}, {Ks[0][0][1][1]:.2f}) [real: ({real_fx:.2f}, {real_fy:.2f})]')

            ########### GT data + GT camera params ############
            R_rel_gt = Rs[0][1] @ torch.inverse(Rs[0][0])
            (kpts1_gt, kpts2_gt), _ = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0][0], ts[0][1], R_rel_gt)
            kpts1_gt = kpts1_gt.reshape((-1, 2))
            kpts2_gt = kpts2_gt.reshape((-1, 2))

            t_rel_gt = -Rs[0][1] @ torch.inverse(Rs[0][0]) @ ts[0][0] + ts[0][1]
            
            dists = distance_between_projections(kpts1_gt, kpts2_gt, Ks[0], Rs[0][0], R_rel_gt, ts[0][0], ts[0][1])
            condition = dists < D
            num_inliers = (condition).sum()

            print(f'Mean distances between corresponding lines (GT all): {dists.mean()}')

            assert(num_inliers == point_corresponds.shape[0])

            inliers = torch.stack((kpts1_gt, kpts2_gt), dim=1)[condition]
            try:
                R_gt1, R_gt2, t_rel, F = find_rotation_matrices(inliers, Ks[0])
            except Exception as ex:
                print(f'[GT data + GT camera params] {ex}')

            scale = (t_rel_gt / t_rel[0]).mean()

            try:
                t_rel = t_rel * scale
                R_gt, t2 = solve_four_solutions(inliers, Ks[0], Rs[0], ts[0], (R_gt1[0], R_gt2[0]), t_rel[0])
            except Exception as ex:
                #print(ex)
                R_sim, m_idx = compare_rotations(Rs, (R_gt1, R_gt2))
                R_gt = R_gt1[0] if m_idx == 0 else R_gt2[0]
                t2 = ts[0][1]
                print('Not all positive (GT data + camera params)')

            kpts_2d_projs, error_2d = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0][0], t2, R_gt)
            error_3d, _ = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0][0], t2, R_gt)

            '''
            # NOTE: These quaternions are used temporarily to compare to trifocal calibration.
            R_rel1_quat = R.from_dcm(R_gt1.cpu()).as_quat()
            R_rel2_quat = R.from_dcm(R_gt2.cpu()).as_quat()
            R_gt_quat = R.from_dcm(R_rel_gt.cpu()).as_quat()
            '''

            R_gt_quat = kornia.rotation_matrix_to_quaternion(R_gt)
            Rs_rel_quat = kornia.rotation_matrix_to_quaternion(R_rel_gt)
            rot_error = torch.mean(torch.abs(R_gt_quat - Rs_rel_quat))

            t_error = torch.norm(t2 - ts[0][1], p=2)
            # TODO: Estimate and evaluate K.
            
            print(f'[GT data + GT camera params]: ({error_2d:.4f}, {error_3d:.4f}), ({rot_error:.4f}, {t_error:.4f})')
            ###################################################

            ########### GT camera params ###########
            dists = distance_between_projections(point_corresponds[:, 0], point_corresponds[:, 1], Ks[0], Rs[0][0], R_rel_gt, ts[0][0], ts[0][1])
            condition = dists < D
            num_inliers = (condition).sum()
            print(f'Number of inliers (GT): {num_inliers} ({P})')
            print(f'Mean distances between corresponding lines (GT): {dists.mean()}')

            inliers = point_corresponds[condition]
            try:
                R_gt1, R_gt2, t_rel, _ = find_rotation_matrices(inliers, Ks[0])
            except Exception as ex:
                print(f'[GT camera params] {ex}')

            scale = (t_rel_gt / t_rel[0]).mean()

            try:
                t_rel = t_rel * scale
                R_gt, t2 = solve_four_solutions(point_corresponds, Ks[0], Rs[0], ts[0], (R_gt1[0], R_gt2[0]), t_rel[0])
            except Exception as ex:
                #print(ex)
                R_sim, m_idx = compare_rotations(Rs, (R_gt1, R_gt2))
                R_gt = R_gt1[0] if m_idx == 0 else R_gt2[0]
                t2 = ts[0][1]
                print('Not all positive (GT camera params)')

            kpts_2d_projs, error_2d = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0][0], t2, R_gt)
            error_3d, kpts_3d_est = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0][0], t2, R_gt)

            R_gt_quat = kornia.rotation_matrix_to_quaternion(R_gt)
            Rs_rel_quat = kornia.rotation_matrix_to_quaternion(R_rel_gt)
            rot_error = torch.mean(torch.abs(R_gt_quat - Rs_rel_quat))

            t_error = torch.norm(t2 - ts[0][1], p=2)

            print(f'[GT camera params]: {error_2d:.4f}, {error_3d:.4f} {rot_error:.4f} {t_error:.4f}')
            ##########################


            ########### Autocalibration using triangulated points ###########
            kpts_2d_projs, _ = evaluate_projection(kpts_3d_est, Ks[0], Rs[0], ts[0][0], t2, R_gt)
            kpts_2d_projs = kpts_2d_projs.reshape(2, -1, 2).transpose(0, 1)

            dists = distance_between_projections(kpts_2d_projs[:, 0], kpts_2d_projs[:, 1], Ks[0], Rs[0][0], R_rel_gt, ts[0][0], t2)
            condition = dists < D
            num_inliers = (condition).sum()

            #assert(num_inliers == point_corresponds.shape[0])

            try:
                R_gt1, R_gt2, t_rel, _ = find_rotation_matrices(kpts_2d_projs, Ks[0])
            except Exception as ex:
                print(f'[Autocalibration using triangulated points (GT)] {ex}')

            scale = (t_rel_gt / t_rel[0]).mean()

            try:
                t_rel = t_rel * scale
                R_gt, t2 = solve_four_solutions(inliers, Ks[0], Rs[0], ts[0], (R_gt1[0], R_gt2[0]), t_rel[0])
            except Exception as ex:
                #print(ex)
                R_sim, m_idx = compare_rotations(Rs, (R_gt1, R_gt2))
                R_gt = R_gt1[0] if m_idx == 0 else R_gt2[0]
                t2 = ts[0][1]
                print('Not all positive (autocalibration using triangulated points)')

            kpts_2d_projs, error_2d = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0][0], t2, R_gt)
            error_3d, _ = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0][0], t2, R_gt)

            R_gt_quat = kornia.rotation_matrix_to_quaternion(R_gt)
            Rs_rel_quat = kornia.rotation_matrix_to_quaternion(R_rel_gt)
            rot_error = torch.mean(torch.abs(R_gt_quat - Rs_rel_quat))

            t_error = torch.norm(t2 - ts[0][1], p=2)
            
            print(f'[Autocalibration using triangulated points (GT)]: {error_2d:.4f}, {error_3d:.4f} {rot_error:.4f} {t_error:.4f}')
            #################################################################

            counter = 0
            line_dist_error_pairs = torch.empty((0, 8), device='cuda', dtype=torch.float32)
            for i in range(N):
                if invalid_counter > 10:
                    if float(invalid_counter) / float(i) > 0.7:
                        invalid_counter = 0
                        break
                    else:
                        intrinsics_found = True

                selected_idxs = torch.tensor(np.random.choice(np.arange(point_corresponds.shape[0]), size=S), device='cuda')

                R_initial1, R_initial2, t_rel, _ = find_rotation_matrices(point_corresponds[selected_idxs], Ks[0])
                
                try:
                    t_rel = t_rel * scale
                    #R_initial, t2 = solve_four_solutions(point_corresponds, Ks[0], Rs[0], ts[0], (R_initial1[0], R_initial2[0]), t_rel[0])
                    R_initial, t2 = solve_four_solutions(point_corresponds, Ks[0], Rs[0], ts[0], (R_initial1[0], R_initial2[0]), None)
                except Exception as ex:
                    #print(ex)
                    # TODO: It's probably OK to just skip these samples.
                    #R_sim, m_idx = compare_rotations(Rs, (R_initial1, R_initial2))
                    #R_initial = R_initial1[0] if m_idx == 0 else R_initial2[0]
                    #t2 = ts[0][1]
                    print('Not all positive')
                    invalid_counter += 1
                    continue

                positive_mask = get_positive_points_mask(point_corresponds, Ks[0], Rs[0], ts[0], R_initial, t2)

                line_dists_initial = distance_between_projections(
                        point_corresponds[:, 0], point_corresponds[:, 1], 
                        Ks[0], Rs[0, 0], R_initial, ts[0][0], t2)

                condition_initial = line_dists_initial < D
                num_inliers_initial = (condition_initial).sum()

                if num_inliers_initial > I:
                    # Evaluate 2D projections and 3D reprojections (triangulation).
                    kpts_2d_projs, error_2d = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0][0], t2, R_initial)
                    error_3d, _ = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0][0], t2, R_initial)

                    if error_3d is None:
                        print('svd_cuda: the updating process of SBDSDC did not converge (error: 3)')
                        invalid_counter += 1
                        continue

                    quaternion_initial = kornia.rotation_matrix_to_quaternion(R_initial)
                    quaternion_gt = kornia.rotation_matrix_to_quaternion(R_gt)
                    quat_norm = torch.norm(quaternion_gt - quaternion_initial, p=1)

                    line_dists_inlier = distance_between_projections(
                        point_corresponds[condition_initial][:, 0], point_corresponds[condition_initial][:, 1], Ks[0], 
                        Rs[0, 0], R_initial, ts[0][0], t2)
                    line_dists_all = distance_between_projections(
                        point_corresponds[:, 0], point_corresponds[:, 1], Ks[0], 
                        Rs[0, 0], R_initial, ts[0][0], t2)
                    
                    line_dist_error_pair = torch.unsqueeze(torch.cat(
                        (quaternion_initial,
                        torch.unsqueeze(num_inliers_initial, dim=0), 
                        torch.unsqueeze(line_dists_all.mean(), dim=0),
                        torch.unsqueeze(error_2d, dim=0),
                        torch.unsqueeze(error_3d, dim=0)), dim=0), dim=0)
                    line_dist_error_pairs = torch.cat((line_dist_error_pairs, line_dist_error_pair), dim=0)
                    print(f'{counter}. ({num_inliers_initial}, {line_dists_inlier.mean():.3f}, {line_dists_all.mean():.3f}) -> {quat_norm:.4f} {error_2d:.2f}, {error_3d:.2f}')

                    counter += 1

            #Ks[0][0][0][0] += K_step_x
            #Ks[0][0][1][1] += K_step_y

            
        def evaluate_top_candidates(quaternions, Ks, Rs, ts, point_corresponds):
            R_rel_quat = torch.mean(quaternions, dim=0)
            R_rel = kornia.quaternion_to_rotation_matrix(R_rel_quat)
            
            kpts_2d_projs, error_2d = evaluate_projection(all_3d_gt, Ks[0], Rs[0], ts[0][0], t2, R_rel)
            error_3d, _ = evaluate_reconstruction(all_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0][0], t2, R_rel)

            R_rel_gt = Rs[0][1] @ torch.inverse(Rs[0][0])
            R_rel_gt_quat = kornia.rotation_matrix_to_quaternion(R_rel_gt)
            rot_error = torch.mean(torch.abs(R_rel_gt_quat - R_rel_quat))

            t_error = torch.norm(t2 - ts[0][1], p=2)

            return rot_error, t_error, error_2d, error_3d


        line_dist_error_pairs_np = line_dist_error_pairs.cpu().numpy()
        top_num_inliers = np.array(sorted(line_dist_error_pairs_np, key=lambda x: x[4]))[:T]
        top_all_dists = np.array(sorted(line_dist_error_pairs_np, key=lambda x: x[5]))[:T]

        top_num_inliers_errors = evaluate_top_candidates(
            torch.tensor(top_num_inliers[:, :4], device='cuda', dtype=torch.float32), Ks, Rs, ts, point_corresponds)
        top_all_dists_errors = evaluate_top_candidates(
            torch.tensor(top_all_dists[:, :4], device='cuda', dtype=torch.float32), Ks, Rs, ts, point_corresponds)

        print(f'Estimated best (num inliers): {line_dist_error_pairs_np[line_dist_error_pairs_np[:, 4].argmax()][[4, 5, 6, 7]]}')
        print(f'Estimated best (all distances): {line_dist_error_pairs_np[line_dist_error_pairs_np[:, 5].argmin()][[4, 5, 6, 7]]}')

        K_error = ((torch.abs(Ks[0][0][0][0] - real_fx) + torch.abs(Ks[0][0][1][1] - real_fy)) / 2) / Ks[0][0][0][2]

        print(f'Error (num inliers top): {top_num_inliers_errors} {K_error}')
        print(f'Error (all distances top): {top_all_dists_errors} {K_error}')

        print(f'Best found: {line_dist_error_pairs_np[line_dist_error_pairs_np[:, 7:].argmin()][4:]}')

        camera_inliers = line_dist_error_pairs_np[:, 3] < 10.

        for idx in range(line_dist_error_pairs_np.shape[1] - 1):
            plt.clf()
            plt.scatter(line_dist_error_pairs_np[~camera_inliers, idx], line_dist_error_pairs_np[~camera_inliers, 3], c='blue')
            plt.scatter(line_dist_error_pairs_np[camera_inliers, idx], line_dist_error_pairs_np[camera_inliers, 3], c='red')
            plt.savefig(f'quat_distro_{idx}.png')
