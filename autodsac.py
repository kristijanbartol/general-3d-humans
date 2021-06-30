# Inspired by: https://github.com/vislearn/DSACLine/blob/master/dsac.py

from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import kornia

from mvn.utils.multiview import find_rotation_matrices, solve_four_solutions, \
    distance_between_projections


class AutoDSAC:
    '''
    Differentiable RANSAC for camera autocalibration.
    '''

    def __init__(self, hyps, sample_size, inlier_thresh, inlier_beta, inlier_alpha, loss_function, scale=None, device='cuda'):
        '''
        Constructor.
        hyps -- number of hypotheses (trials) for each AutoDSAC iteration
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
        self.loss_function = loss_function
        self.device = device

        # Rotation and translation for the reference camera.
        self.R_ref = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], 
            dtype=torch.float32, device=self.device)
        self.t_ref = torch.tensor([0., 0., 0.], dtype=torch.float32, device=self.device).transpose((0, 1))

        # The scale is known.
        # TODO: Currently, working without scale (None) to learn rotations only.
        self.scale = scale

    def __sample_hyp(self, point_corresponds, Ks) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Select a random subset of point correspondences and calculate R and t.

        point_corresponds --- Px2x2 (P=M*J)
        Ks --- 2x3x3, GT or estimated intrinsics for cam1 and cam2
        '''
        selected_idxs = torch.tensor(np.random.choice(
            np.arange(point_corresponds.shape[0]), size=self.hyps), device='cuda')

        R_est1, R_est2, t_rel, _ = find_rotation_matrices(
            point_corresponds[selected_idxs], None, Ks)

        t_rel = t_rel * self.scale
        try:
            R_est, t_rel = solve_four_solutions(
                point_corresponds, Ks[0], self.R_ref, self.t_ref, (R_est1[0], R_est2[0]), t_rel)
        except Exception as ex:
            # In case none of the four solutions has all positive points.
            return None

        return R_est, t_rel

    def __soft_inlier_count(self, point_corresponds, R_est, t_est, Ks):
        '''
        Soft inlier count for given camera parameters and point correspondences.

        point_corresponds --- Px2x2 (P=M*J)
        Ks --- 2x3x3, GT or estimated intrinsics
        '''

        # 3D line distances.
        line_dists = distance_between_projections(
            point_corresponds[:, 0], point_corresponds[:, 1], 
            Ks[0], self.R_ref, R_est, self.t_ref, t_est)

        # Soft inliers.
        line_dists = 1 - torch.sigmoid(self.inlier_beta *
                                  (line_dists - self.inlier_thresh))
        score = torch.sum(line_dists)

        return score, line_dists

    def __call__(self, point_corresponds, Ks, Rs, ts):
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

        # Working on CPU because of many small matrices.
        point_corresponds = point_corresponds.cpu()

        num_frames: int = point_corresponds.size(0)

        avg_exp_loss: float = 0                            # expected loss
        hyp_losses = torch.zeros([self.hyps, 1])    # loss of each hypothesis
        hyp_scores = torch.zeros([self.hyps, 1])    # score of each hypothesis
        max_score = 0 	                            # score of best hypothesis

        for h in range(0, self.hyps):

            # === Step 1: Sample hypothesis ===========================
            cam_params = self.__sample_hyp(point_corresponds, Ks)
            if cam_params is None:
                continue  # skip invalid hyps

            # === Step 2: Score hypothesis using soft inlier count ====
            score, _ = self.__soft_inlier_count(
                point_corresponds, cam_params[0], cam_params[1], Ks)

            # === Step 3: Calculate loss of hypothesis ================
            # TODO: For now, GT is only rotation (QuaternionLoss).
            R_rel_gt, _ = kornia.relative_camera_motion(Rs[0], ts[0], Rs[1], ts[1])
            loss = self.loss_function(cam_params[0], R_rel_gt)

            # Store results.
            hyp_losses[h] = loss
            hyp_scores[h] = score

            # Keep track of best hypothesis so far.
            if score > max_score:
                max_score = score
                best_loss = loss
                best_params = cam_params

            # === Step 4: calculate the expectation ===========================

            # Softmax distribution from hypotheses scores.
            hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

            # Loss expectation.
            exp_loss = torch.sum(hyp_losses * hyp_scores)
            avg_exp_loss += exp_loss

        return best_params, avg_exp_loss / num_frames, best_loss
