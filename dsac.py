# Inspired by: https://github.com/vislearn/DSACLine/blob/master/dsac.py

from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import itertools

from mvn.utils.multiview import find_rotation_matrices, solve_four_solutions, \
    distance_between_projections, triangulate_point_from_multiple_views_linear_torch
from mvn.utils.vis import CONNECTIVITY_DICT
from metrics import rel_mpjpe


class DSAC:
    '''Abstract DSAC class.
    
    '''
    def __init__(self):
        pass


class CameraDSAC(DSAC):
    '''
    Differentiable RANSAC for camera autocalibration.
    '''

    def __init__(self, hyps, sample_size, inlier_thresh, inlier_beta, entropy_beta, min_entropy, entropy_to_scores,
            temp, gumbel, hard, score_nn, loss_function, scale=None, device='cpu'):
        '''
        Constructor.
        hyps -- number of hypotheses (trials) for each CameraDSAC iteration
        sample_size -- number of point correspondences to use for camera parameter estimation
        inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
        inlier_beta -- scaling factor within the sigmoid of the soft inlier count
        loss_function -- function to compute the quality of estimated line parameters wrt ground truth
        scale --- scalar, GT scale, used to obtain proper translation
        device --- 'cuda' or 'cpu'
        '''

        self.hyps = hyps
        self.sample_size = sample_size
        self.inlier_thresh = inlier_thresh
        self.inlier_beta = inlier_beta
        
        self.temp = temp

        self.entropy_beta = entropy_beta
        self.min_entropy = min_entropy
        self.entropy_to_scores = entropy_to_scores

        self.gumbel = gumbel
        self.hard = hard

        self.score_nn = score_nn
        self.loss_function = loss_function
        self.device = device

        # Rotation and translation for the reference camera.
        # TODO: Currently not using these reference rotation and translation.
        self.R_ref = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], 
            dtype=torch.float32, device=self.device)
        self.t_ref = torch.tensor([[0., 0., 0.]], dtype=torch.float32, device=self.device).transpose(0, 1)

        # The scale is known.
        # TODO: Currently, working without scale (None) to learn rotations only.
        self.scale = scale

    def __sample_hyp(self, point_corresponds, Ks, Rs, ts) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Select a random subset of point correspondences and calculate R and t.

        point_corresponds --- Px2x2 (P=M*J)
        Ks --- 2x3x3, GT or estimated intrinsics for cam1 and cam2
        '''
        #selected_idxs = torch.tensor(np.random.choice(
        #    np.arange(point_corresponds.shape[0]), size=self.sample_size), device=self.device)
        selected_idxs = np.random.choice(
            np.arange(point_corresponds.shape[0]), size=self.sample_size)

        R_est1, R_est2, t_rel, _ = find_rotation_matrices(
            point_corresponds[selected_idxs], Ks, device=self.device)

        #t_rel = t_rel * self.scale
        try:
            R_est, _ = solve_four_solutions(
                point_corresponds, Ks, Rs, ts, (R_est1[0], R_est2[0]), None)
        except Exception as ex:
            # In case none of the four solutions has all positive points.
            return None

        return R_est, t_rel

    def __score_nn(self, point_corresponds, R_est, t_est, Ks, Rs, ts):
        '''
        Feed 3D line distances into ScoreNN to obtain score for the hyp.

        point_corresponds -- Px2x2
        R_est -- 3x3, estimated relative rotation
        t_est -- 3x1, estimated relative translation
        Rs -- GT rotations for the first and second camera
        ts -- GT translation for the first and second camera
        '''
        #line_dists = distance_between_projections(
        #    point_corresponds[:, 0], point_corresponds[:, 1], 
        #    Ks[0], Rs[0, 0], R_est, ts[0, 0], t_est[0], device=self.device)
        line_dists = distance_between_projections(
            point_corresponds[:, 0], point_corresponds[:, 1], 
            Ks, Rs[0], R_est, ts[0], ts[1], device=self.device)

        # Normalize and invert line distance values for NN.
        #model_input = line_dists / line_dists.max()
        #model_input = 1 - model_input
        line_dists, _ = torch.sort(line_dists, dim=0, descending=True)

        #return 1 - self.score_nn(line_dists.cuda()), line_dists.mean()
        return self.score_nn(line_dists.cuda()), line_dists.mean()

    @staticmethod
    def __get_absolute_params(R_init, t_init, R_rel, t_rel):
        '''Calculate absolute params based on init and relative params.
        
        Parameters:
        -----------
        R_init -- [3x3]
        t_init -- [3x1]
        R_rel -- [3x3]
        ts -- [3x1]

        Return:
        -------
        abs_Rs -- [3x3]
        abs_ts -- [3x1]
        '''
        return R_rel @ R_init, t_init + t_rel

    def __call__(self, point_corresponds, Ks, Rs, ts, points_3d):
        '''
        Perform robust, differentiable autocalibration.

        Returns the expected loss of choosing a good camera params hypothesis, used for backprop.
        Labels are used to calculate 3D reprojection loss. Another possibility is to compare
        rotations and translations.
        point_corresponds -- predicted 2D points for pairs of images, array of shape (Mx2x2) where
                M is the number of frames
                2 is the number of views
                2 is the number of coordinates (x, y)
        gt_3d -- ground truth labels for the set of frames, array of shape (MxJx3) where
                M is the number of frames
                J is the number of joints
                3 is the number of coordinates (x, y, z)
        '''
        hyp_losses = torch.ones([self.hyps, 1], device=self.device) * 100.        # loss of each hypothesis
        hyp_scores = torch.zeros([self.hyps, 1], device=self.device)              # score of each hypothesis
        line_dists = torch.ones([self.hyps, 1], device=self.device) * 1000.       # list of all line dists
        
        #Rs_hyps = torch.zeros([self.hyps, 3, 3], device=self.device)
        #ts_hyps = torch.zeros([self.hyps, 3, 1], device=self.device)

        max_score = 0.
        min_line_dist = 100.
        selected_params = None
        invalid_hyps = 0

        for h in range(0, self.hyps):

            # === Step 1: Sample hypothesis ===========================
            cam_params = self.__sample_hyp(point_corresponds, Ks, Rs, ts)
            if cam_params is None:
                invalid_hyps += 1
                continue  # skip invalid hyps

            # === Step 2: Score hypothesis using soft inlier count ====
            score, line_dist = self.__score_nn(
                point_corresponds, cam_params[0], cam_params[1], Ks, Rs, ts)

            # === Step 3: Calculate loss of hypothesis ================
            loss = self.loss_function(cam_params[0], Ks, Rs, ts, points_3d)

            # Store results.
            hyp_losses[h] = loss
            hyp_scores[h] = score
            line_dists[h] = line_dist
            #Rs_hyps[h] = cam_params[0]
            #ts_hyps[h] = cam_params[1]

            if score > max_score:
                max_score = score
                selected_params = cam_params

            if line_dist < min_line_dist:
                min_line_dist = line_dist
                best_line_dist_hyp = cam_params

        # === Step 4: calculate the expectation ===========================

        # Softmax distribution from hypotheses scores.
        if self.gumbel:
            #softmax_mask = torch.zeros((hyp_scores.shape[0], 1), dtype=torch.float32, device=self.device)
            #softmax_mask[hyp_scores.nonzero(as_tuple=True)[0]] = 1.
            hyp_scores_softmax = F.gumbel_softmax(hyp_scores, tau=self.temp, hard=self.hard, dim=0)
        else:  
            hyp_scores_softmax = F.softmax(hyp_scores / self.temp, dim=0)

        # Store best hypotheses (for logging).
        best_loss_idx = torch.argmin(hyp_losses, dim=0)
        best_softmax_score_idx = torch.argmax(hyp_scores_softmax, dim=0)
        best_score_idx = torch.argmax(hyp_scores, dim=0)
        best_line_dist_idx = torch.argmin(line_dists, dim=0)

        best_loss = (hyp_losses[best_loss_idx], hyp_scores_softmax[best_loss_idx], 
            hyp_scores[best_loss_idx], line_dists[best_loss_idx])
        best_softmax_score = (hyp_losses[best_softmax_score_idx], hyp_scores_softmax[best_softmax_score_idx], 
            hyp_scores[best_softmax_score_idx], line_dists[best_softmax_score_idx])
        best_score = (hyp_losses[best_score_idx], hyp_scores_softmax[best_score_idx], 
            hyp_scores[best_score_idx], line_dists[best_score_idx])
        best_line_dist = (hyp_losses[best_line_dist_idx], hyp_scores_softmax[best_line_dist_idx], 
            hyp_scores[best_line_dist_idx], line_dists[best_line_dist_idx])

        # Loss expectation.
        if self.entropy_to_scores:
            softmax_entropy = -torch.sum(hyp_scores * torch.log(hyp_scores_softmax))
            #softmax_entropy = -torch.sum(hyp_scores * torch.log(hyp_scores))
        else:
            softmax_entropy = -torch.sum(hyp_scores_softmax * torch.log(hyp_scores_softmax))

        hyp_losses /= hyp_losses.max()

        exp_loss = torch.sum(hyp_losses * hyp_scores_softmax)
        entropy_loss = max(0., (softmax_entropy - self.min_entropy))

        total_loss = exp_loss + self.entropy_beta * softmax_entropy

        if not invalid_hyps == self.hyps:
            selected_params = self.__get_absolute_params(
                Rs[0], ts[0], selected_params[0], selected_params[1])
            best_line_dist_hyp = self.__get_absolute_params(
                Rs[0], ts[0], best_line_dist_hyp[0], best_line_dist_hyp[1])
        else:
            print('All scores are zero!')
            return None

        print(f'Invalid hyps: {invalid_hyps}')

        return total_loss, exp_loss, entropy_loss, selected_params, best_loss, best_softmax_score, best_score, best_line_dist, best_line_dist_hyp



class PoseDSAC(DSAC):
    '''
    Differentiable RANSAC for pose triangulation.
    '''

    def __init__(self, hyps, num_joints, entropy_beta, min_entropy, entropy_to_scores,
            temp, gumbel, hard, body_lengths_mode, weighted_selection, weighted_beta,
            score_nn, loss_function, scale=None, device='cpu'):
        '''
        Constructor.
        hyps -- number of hypotheses (trials) for each PoseDSAC iteration
        loss_function -- function to estimate the quality of triangulated poses
        scale --- scalar, GT scale, used to obtain proper translation
        device --- 'cuda' or 'cpu'
        '''
        self.hyps = hyps
        self.num_joints = num_joints

        self.entropy_beta = entropy_beta
        self.min_entropy = min_entropy
        self.temp = temp
        self.weighted_beta = weighted_beta
        
        self.entropy_to_scores = entropy_to_scores
        self.gumbel = gumbel
        self.hard = hard
        self.weighted_selection = weighted_selection

        self.body_lengths_mode = body_lengths_mode

        self.score_nn = score_nn
        self.loss_function = loss_function
        self.device = device

    @staticmethod
    def __prepare_projection_matrix(K, R, t):
        return K @ torch.cat((R, t), dim=1)

    def __triangulate_joint(self, est_2d_pose, joint_idx, Ks, Rs, ts, cidxs):
        Ps = torch.stack(
            [self.__prepare_projection_matrix(Ks[x], Rs[x], ts[x]) for x in cidxs], 
            dim=0
        )
        points_2d = torch.stack([est_2d_pose[x] for x in cidxs], dim=0)[:, joint_idx]

        return triangulate_point_from_multiple_views_linear_torch(Ps, points_2d)

    def __sample_hyp(self, est_2d_pose, Ks, Rs, ts, baseline=False):
        '''
        Select a random subset of point correspondences and calculate R and t.

        est_2d_pose --- CxJx2
        Ks --- Cx3x3, GT or estimated intrinsics
        Rs --- Cx3x3, GT or estimated intrinsics
        ts --- Cx3x3, GT or estimated intrinsics
        '''
        num_cameras = est_2d_pose.shape[0]
        num_joints = est_2d_pose.shape[1]

        # Select indexes for view combination subsets.
        all_view_combinations = []
        for l in range(2, num_cameras + 1):
        #for l in range(3, num_cameras + 1):
            all_view_combinations += list(itertools.combinations(list(range(num_cameras)), l))
        selected_combination_idxs = np.random.choice(
            np.arange(len(all_view_combinations)), size=num_joints,
            p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.075, 0.075, 0.075, 0.075, 0.4])
            #p=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.54])

        # For each joint, use the selected view subsets to triangulate points.
        pose_3d = torch.zeros([num_joints, 3], dtype=torch.float32, device=self.device)
        baseline_pose = torch.zeros([num_joints, 3], dtype=torch.float32, device=self.device) \
            if baseline else None

        for joint_idx in range(num_joints):
            cidxs = all_view_combinations[selected_combination_idxs[joint_idx]]
            all_views_cidxs = all_view_combinations[-1]

            pose_3d[joint_idx] = self.__triangulate_joint(
                est_2d_pose, joint_idx, Ks, Rs, ts, cidxs
            )
            # Do not calculate baseline more than once for efficiency.
            if baseline:
                baseline_pose[joint_idx] = self.__triangulate_joint(
                    est_2d_pose, joint_idx, Ks, Rs, ts, all_views_cidxs
                )

        return pose_3d, baseline_pose

    def __score_nn(self, est_3d_pose, mean, std):
        '''
        Feed 3D pose coordinates into ScoreNN to obtain score for the hyp.

        est_3d_pose
        '''
        # Standardize pose.
        est_3d_pose_norm = ((est_3d_pose - mean) / std)

        # Zero-center around hip joint.
        est_3d_pose_norm = est_3d_pose_norm - est_3d_pose_norm[0]

        # Extract body part lengths.
        if self.body_lengths_mode == 1 or self.body_lengths_mode == 2:
            connections = CONNECTIVITY_DICT['human36m']
            lengths = []
            for (kpt1, kpt2) in connections:
                lengths.append(torch.norm(est_3d_pose_norm[kpt1] - est_3d_pose_norm[kpt2]))
            lengths = torch.stack(lengths, dim=0)

        # Select network input based on body lengths mode.
        if self.body_lengths_mode == 0:
            network_input = est_3d_pose_norm.flatten()
        elif self.body_lengths_mode == 1:
            network_input = torch.cat((est_3d_pose_norm.flatten(), lengths), dim=0)
        elif self.body_lengths_mode == 2:
            network_input = lengths

        return self.score_nn(network_input.cuda())

    def __call__(self, est_2d_pose, Ks, Rs, ts, gt_3d, mean, std):
        '''
        Perform robust, differentiable triangulation.

        est_2d_pose -- predicted 2D points for B frames: (CxJx2)
        Ks -- GT/estimated intrinsics for B frames: (Cx3x3)
        Rs -- GT/estimated rotations for B frames: (Cx3x3)
        ts -- GT/estimated translations for B frames: (Cx3x1)
        gts_3d -- GT 3D poses: (Jx3)
        '''
        hyp_losses = torch.zeros([self.hyps, 1], device=self.device)                   # hyp losses
        hyp_scores = torch.zeros([self.hyps, 1], device=self.device)                   # hyp scores
        hyps_3d = torch.zeros([self.hyps, self.num_joints, 3], device=self.device)
        
        best_loss_loss = 1000           # this one is a reference
        best_loss_score = 0

        best_score_loss = 0
        best_score_score = 0            # this one is a reference

        baseline = None
        selected_pose = None

        for h in range(0, self.hyps):

            # === Step 1: Sample hypothesis ===========================
            calculate_baseline = True if baseline is None else False
            sample_tuple = self.__sample_hyp(est_2d_pose, Ks, Rs, ts, calculate_baseline)
            sample = sample_tuple[0]

            # === Step 2: Score hypothesis using soft inlier count ====
            score = self.__score_nn(sample, mean, std)

            # === Step 3: Calculate loss of hypothesis ================
            if baseline is None:
                baseline = sample_tuple[1]
                baseline_loss = self.loss_function(baseline, gt_3d)
            loss = self.loss_function(sample, gt_3d)

            # Store results.
            hyp_losses[h] = loss
            hyp_scores[h] = score
            hyps_3d[h] = sample

            # Keep track of best hypotheses with respect to loss and score.
            if loss < best_loss_loss:
                best_loss_loss = loss
                best_loss_score = score

            if score > best_score_score:
                best_score_loss = loss
                best_score_score = score
                selected_pose = sample
        
        best_loss = (best_loss_loss, best_loss_score)
        best_score = (best_score_loss, best_score_score)

        # === Step 4: calculate the expectation ===========================

        # Softmax distribution from hypotheses scores.
        if self.gumbel:
            hyp_scores_softmax = F.gumbel_softmax(hyp_scores, tau=self.temp, hard=self.hard, dim=0)
        else:  
            hyp_scores_softmax = F.softmax(hyp_scores / self.temp, dim=0)

        # Calculate metrics.
        hyp_losses_sorted, _ = torch.sort(hyp_losses, dim=0)
        hyp_rank = (hyp_losses_sorted == best_score_loss).nonzero(as_tuple=True)[0].float()
        hyp_scores_sorted, hyp_scores_sorted_idxs = torch.sort(hyp_scores, dim=0, descending=True)
        hyp_losses_sorted_by_scores = hyp_losses[hyp_scores_sorted_idxs[:, 0]]
        top_loss = hyp_losses_sorted_by_scores[:5].mean()
        bottom_loss = hyp_losses_sorted_by_scores[-5:].mean()

        # Weighted selection.
        final_pose_avg = torch.zeros((self.num_joints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.hyps):
            #final_pose += hyps_3d[hidx] * hyp_scores[hidx, 0]
            final_pose_avg += hyps_3d[hyp_scores_sorted_idxs[:, 0]][hidx] * 1.0
        final_pose_avg /= self.hyps
        avg_pose_loss = self.loss_function(final_pose_avg, gt_3d)

        final_pose = torch.zeros((self.num_joints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.hyps):
            #final_pose += hyps_3d[hidx] * hyp_scores[hidx, 0]
            final_pose += hyps_3d[hyp_scores_sorted_idxs[:, 0]][hidx] * hyp_scores_sorted[hidx, 0]
        final_pose /= hyp_scores_sorted.sum()
        weighted_loss = self.loss_function(final_pose, gt_3d)

        # Random pose.
        random_pose = hyps_3d[torch.randint(self.hyps, (1,))[0]]
        random_pose_loss = self.loss_function(random_pose, gt_3d)

        if self.weighted_selection:
            return weighted_loss, best_loss
        else:
            # Entropy loss.
            if self.entropy_to_scores:
                #softmax_entropy = -torch.sum(hyp_scores * torch.log(hyp_scores))
                softmax_entropy = -torch.sum(hyp_scores * torch.log(hyp_scores_softmax))
            else:
                softmax_entropy = -torch.sum(hyp_scores_softmax * torch.log(hyp_scores_softmax))

            # Loss expectation.
            hyp_losses /= hyp_losses.max()

            exp_loss = torch.sum(hyp_losses * hyp_scores_softmax)
            entropy_loss = max(0., (softmax_entropy - self.min_entropy))

            total_loss = exp_loss + self.entropy_beta * softmax_entropy + self.weighted_beta * weighted_loss

            return total_loss, exp_loss, entropy_loss, baseline_loss, weighted_loss, avg_pose_loss, random_pose_loss, selected_pose, best_loss, best_score, hyp_rank, top_loss, bottom_loss
