# Inspired by: https://github.com/vislearn/DSACLine/blob/master/dsac.py

from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import itertools

from mvn.utils.multiview import find_rotation_matrices, solve_four_solutions, \
    distance_between_projections, triangulate_point_from_multiple_views_linear_torch
from loss import cross_entropy_loss


class DSAC:
    '''Abstract DSAC class.
    
    '''
    def __init__(self):
        pass


class CameraDSAC(DSAC):
    '''
    Differentiable RANSAC for camera autocalibration.
    '''

    def __init__(self, hyps, sample_size, inlier_thresh, inlier_beta, inlier_alpha, score_nn, 
            loss_function, scale=None, device='cpu'):
        '''
        Constructor.
        hyps -- number of hypotheses (trials) for each CameraDSAC iteration
        sample_size -- number of point correspondences to use for camera parameter estimation
        inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
        inlier_beta -- scaling factor within the sigmoid of the soft inlier count
        inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
        loss_function -- function to compute the quality of estimated line parameters wrt ground truth
        scale --- scalar, GT scale, used to obtain proper translation
        device --- 'cuda' or 'cpu'
        '''

        self.hyps = hyps
        self.sample_size = sample_size
        self.inlier_thresh = inlier_thresh
        self.inlier_beta = inlier_beta
        self.inlier_alpha = inlier_alpha
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
        hyp_line_dists = torch.zeros([self.hyps, 1], device=self.device)    # line dists of each hypothesis
        hyp_losses = torch.zeros([self.hyps, 1], device=self.device)        # loss of each hypothesis
        hyp_scores = torch.zeros([self.hyps, 1], device=self.device)        # score of each hypothesis

        best_loss_loss = 1000           # this one is a reference
        best_loss_score = 0
        best_loss_line_dist = 0

        best_score_loss = 0
        best_score_score = 0            # this one is a reference
        best_score_line_dist = 0

        best_line_dist_loss = 0
        best_line_dist_score = 0
        best_line_dist_dist = 10000     # this one is a reference

        selected_params = None

        for h in range(0, self.hyps):

            # === Step 1: Sample hypothesis ===========================
            cam_params = self.__sample_hyp(point_corresponds, Ks, Rs, ts)
            if cam_params is None:
                continue  # skip invalid hyps

            # === Step 2: Score hypothesis using soft inlier count ====
            score, line_dist = self.__score_nn(
                point_corresponds, cam_params[0], cam_params[1], Ks, Rs, ts)

            # === Step 3: Calculate loss of hypothesis ================
            loss = self.loss_function(cam_params[0], Ks, Rs, ts, points_3d)

            # Store results.
            hyp_losses[h] = loss
            hyp_scores[h] = score
            hyp_line_dists[h] = line_dist

            #print(f'{h}. {loss.item():3.4f} \t{score.item():3.4f} \t{line_dist.item():3.4f}')

            # Keep track of best hypotheses with respect to loss, score and line dists.
            # TODO: Simplify this by taking argmax for each reference variable.
            if loss < best_loss_loss:
                best_loss_loss = loss
                best_loss_score = score
                best_loss_line_dist = line_dist

            if score > best_score_score:
                best_score_loss = loss
                best_score_score = score
                best_score_line_dist = line_dist
                selected_params = cam_params

            if line_dist < best_line_dist_dist:
                best_line_dist_loss = loss
                best_line_dist_score = score
                best_line_dist_dist = line_dist

        best_loss = (best_loss_loss, best_loss_score, best_loss_line_dist)
        best_score = (best_score_loss, best_score_score, best_score_line_dist)
        best_line_dist = (best_line_dist_loss, best_line_dist_score, best_line_dist_dist)

        # TODO: Calculate correlations

        # === Step 4: calculate the expectation ===========================

        # Softmax distribution from hypotheses scores.
        #hyp_scores_softmax = F.softmax(self.inlier_alpha * hyp_scores)
        hyp_scores_softmax = F.softmax(hyp_scores, dim=0)

        # Loss expectation.
        hyp_losses /= hyp_losses.max()
        softmax_entropy = -torch.sum(hyp_scores_softmax * torch.log(hyp_scores_softmax))

        exp_loss = torch.sum(hyp_losses * hyp_scores_softmax) + 0.6 * softmax_entropy #* 2 * (0.3 / 5.3)
        #exp_loss = torch.sum(hyp_losses * hyp_scores_softmax)

        # Categorical cross-entropy loss.
        cross_entropy = cross_entropy_loss(hyp_scores_softmax, hyp_losses)

        selected_params = self.__get_absolute_params(
            Rs[0], ts[0], selected_params[0], selected_params[1])

        return exp_loss, softmax_entropy, selected_params, best_loss, best_score, best_line_dist



class PoseDSAC(DSAC):
    '''
    Differentiable RANSAC for pose triangulation.
    '''

    def __init__(self, hyps, score_nn, loss_function, device='cpu'):
        '''
        Constructor.
        hyps -- number of hypotheses (trials) for each PoseDSAC iteration
        loss_function -- function to estimate the quality of triangulated poses
        scale --- scalar, GT scale, used to obtain proper translation
        device --- 'cuda' or 'cpu'
        '''
        self.hyps = hyps
        self.score_nn = score_nn
        self.loss_function = loss_function
        self.device = device

    @staticmethod
    def __prepare_projection_matrix(K, R, t):
        return K @ torch.cat((R, t), dim=0)

    def __sample_hyp(self, est_2d_pose, Ks, Rs, ts) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Select a random subset of point correspondences and calculate R and t.

        est_2d_pose --- CxJx2
        Ks --- Cx3x3, GT or estimated intrinsics
        Rs --- Cx3x3, GT or estimated intrinsics
        ts --- Cx3x3, GT or estimated intrinsics
        '''
        num_cameras = est_2d_pose.shape[0]
        num_joints = est_2d_pose.shape[1]

        all_view_combinations = []
        for l in range(2, num_cameras):
            all_view_combinations += itertools.combinations(list(range(num_cameras)))
        selected_combinations = np.random.choice(
            np.arange(len(all_view_combinations)), size=num_joints)

        pose_3d = torch.zeros([num_joints, 3], dtype=torch.float32, device=self.device)
        for joint_idx in range(num_joints):
            cidxs = selected_combinations[joint_idx]
            Ps = torch.cat(
                [self.__prepare_projection_matrix(Ks[x], Rs[x], ts[x]) for x in cidxs], 
                dim=0
            )
            points_2d = est_2d_pose[cidxs]

            # TODO: Pass confidences here when NG-RANSAC is implemented.
            pose_3d[joint_idx] = \
                triangulate_point_from_multiple_views_linear_torch(Ps, points_2d)

        return pose_3d

    def __score_nn(self, est_3d_pose):
        '''
        Feed 3D pose coordinates into ScoreNN to obtain score for the hyp.

        est_3d_pose
        '''

        # Normalize and invert line distance values for NN.
        #model_input = line_dists / line_dists.max()
        #model_input = 1 - model_input

        return self.score_nn(est_3d_pose)

    # TODO: Update this to work in parallel for all batches.
    def __call__(self, est_2d_pose, Ks, Rs, ts, gt_3d):
        '''
        Perform robust, differentiable triangulation.

        est_2d_pose -- predicted 2D points for B frames: (CxJx2)
        Ks -- GT/estimated intrinsics for B frames: (Cx3x3)
        Rs -- GT/estimated rotations for B frames: (Cx3x3)
        ts -- GT/estimated translations for B frames: (Cx3x1)
        gts_3d -- GT 3D poses: (Jx3)
        '''
        hyp_poses = torch.zeros([self.hyps, gt_3d.shape[0], 3], device=self.device)    # pose hyps
        hyp_losses = torch.zeros([self.hyps, 1], device=self.device)                   # hyp losses
        hyp_scores = torch.zeros([self.hyps, 1], device=self.device)                   # hyp scores

        '''
        best_loss_loss = 1000           # this one is a reference
        best_loss_score = 0
        best_loss_line_dist = 0

        best_score_loss = 0
        best_score_score = 0            # this one is a reference
        best_score_line_dist = 0

        best_line_dist_loss = 0
        best_line_dist_score = 0
        best_line_dist_dist = 10000     # this one is a reference
        '''
        best_score = 0.

        selected_pose = None

        for h in range(0, self.hyps):

            # === Step 1: Sample hypothesis ===========================
            triang_3d = self.__sample_hyp(est_2d_pose, Ks, Rs, ts)

            # === Step 2: Score hypothesis using soft inlier count ====
            score = self.__score_nn(triang_3d)

            # === Step 3: Calculate loss of hypothesis ================
            loss = self.loss_function(triang_3d, gt_3d)

            # Store results.
            hyp_poses[h] = h
            hyp_losses[h] = loss
            hyp_scores[h] = score

            #print(f'{h}. {loss.item():3.4f} \t{score.item():3.4f} \t{line_dist.item():3.4f}')

            '''
            # Keep track of best hypotheses with respect to loss, score and line dists.
            # TODO: Simplify this by taking argmax for each reference variable.
            if loss < best_loss_loss:
                best_loss_loss = loss
                best_loss_score = score
                best_loss_line_dist = line_dist

            if score > best_score_score:
                best_score_loss = loss
                best_score_score = score
                best_score_line_dist = line_dist
                selected_pose = triang_3d

            if line_dist < best_line_dist_dist:
                best_line_dist_loss = loss
                best_line_dist_score = score
                best_line_dist_dist = line_dist
            '''
            if score > best_score:
                best_score = score
                selected_pose = triang_3d
        '''
        best_loss = (best_loss_loss, best_loss_score, best_loss_line_dist)
        best_score = (best_score_loss, best_score_score, best_score_line_dist)
        best_line_dist = (best_line_dist_loss, best_line_dist_score, best_line_dist_dist)
        '''

        # === Step 4: calculate the expectation ===========================

        # Softmax distribution from hypotheses scores.
        #hyp_scores_softmax = F.softmax(self.inlier_alpha * hyp_scores)
        hyp_scores_softmax = F.softmax(hyp_scores, dim=0)

        # Loss expectation.
        hyp_losses /= hyp_losses.max()
        softmax_entropy = -torch.sum(hyp_scores_softmax * torch.log(hyp_scores_softmax))

        exp_loss = torch.sum(hyp_losses * hyp_scores_softmax)

        # Categorical cross-entropy loss.
        #cross_entropy = cross_entropy_loss(hyp_scores_softmax, hyp_losses)

        return exp_loss, selected_pose
