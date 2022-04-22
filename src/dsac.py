# Author: Kristijan Bartol
# Inspired by: https://github.com/vislearn/DSACLine/blob/master/dsac.py


from typing import Callable, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from kornia.geometry.conversions import angle_axis_to_rotation_matrix

from mvn.utils.multiview import find_rotation_matrices, solve_four_solutions, \
    distance_between_projections, triangulate_point_from_multiple_views_linear_torch
from mvn.utils.vis import CONNECTIVITY_DICT, KPTS
from metrics import center_pelvis, rel_mpjpe
from hypothesis import CameraHypothesisPool, CameraParams, HypothesisPool
from abstract import DSAC, LossFunction


class CameraDSAC(DSAC):
    '''
    Differentiable RANSAC for camera autocalibration.
    '''

    def __init__(
        self, 
        hyps: int, 
        sample_size: int, 
        inlier_thresh: float, 
        inlier_beta: float, 
        entropy_beta: float, 
        entropy_to_scores: bool,
        temp: float, 
        gumbel: bool, 
        hard: bool, 
        exp_beta: float, 
        est_beta: float, 
        score_nn: Callable, 
        loss_function: LossFunction, 
        scale: Optional[float] = None, 
        device: str = 'cpu'
    ) -> None:
        ''' CameraDSAC constructor.
        
            Parameters
            ----------
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

        self.exp_beta = exp_beta
        self.entropy_beta = entropy_beta
        self.est_beta = est_beta
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

    def __score_nn(self, point_corresponds, R_est, Ks, Rs, ts):
        '''
        Feed 3D line distances into ScoreNN to obtain score for the hyp.

        point_corresponds -- Px2x2
        R_est -- 3x3, estimated relative rotation
        t_est -- 3x1, estimated relative translation
        Rs -- GT rotations for the first and second camera
        ts -- GT translation for the first and second camera
        '''
        line_dists = distance_between_projections(
            point_corresponds[:, 0], point_corresponds[:, 1], 
            Ks, Rs[0], R_est[0], ts[0], ts[1], device=self.device)

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

    def __call__(self, point_corresponds, gt_Ks, gt_Rs, gt_ts, points_3d, metrics):
        ''' Perform robust, differentiable autocalibration.

            Returns the expected loss of choosing a good camera params hypothesis, used for backprop.
            Labels are used to calculate 3D reprojection loss. Another possibility is to compare
            rotations and translations.
        
            Parameters
            ----------
            point_corresponds -- predicted 2D points for pairs of images, array of shape (Mx2x2) where
                    M is the number of frames
                    2 is the number of views
                    2 is the number of coordinates (x, y)
            gt_3d -- ground truth labels for the set of frames, array of shape (MxJx3) where
                    M is the number of frames
                    J is the number of joints
                    3 is the number of coordinates (x, y, z)
        '''
        
        gt_params = CameraParams(gt_Ks[0], gt_Ks[1], gt_Rs[0], gt_ts[0], 
            CameraParams.to_R_rel(gt_Rs[0], gt_Rs[1]),
            CameraParams.to_t_rel(gt_ts[0], gt_ts[1], gt_Rs[0], gt_Rs[1]))
        hpool = CameraHypothesisPool(self.hyps, gt_params, points_3d, self.loss_function, device=self.device)

        hyp_idx = 0
        while hyp_idx < self.hyps:
            cam_params = self.__sample_hyp(point_corresponds, gt_Ks, gt_Rs, gt_ts)
            if cam_params is None:
                continue    # skip invalid hyps
            else:
                hyp_idx += 1

            score = self.__score_nn(point_corresponds, cam_params, gt_Ks, gt_Rs, gt_ts)
            hpool.append(cam_params, score)

        # Softmax distribution from hypotheses scores.
        if self.gumbel:
            hyp_scores_softmax = F.gumbel_softmax(hpool.scores, tau=self.temp, hard=self.hard, dim=0)
        else:  
            hyp_scores_softmax = F.softmax(hpool.scores / self.temp, dim=0)

        # Entropy loss.
        if self.entropy_to_scores:
            #softmax_entropy = -torch.sum(hyp_scores * torch.log(hyp_scores))
            softmax_entropy = -torch.sum(hpool.scores * torch.log(hyp_scores_softmax))
        else:
            softmax_entropy = -torch.sum(hyp_scores_softmax * torch.log(hyp_scores_softmax))

        # Expectation loss.
        hpool.losses /= hpool.losses.max()
        exp_loss = torch.sum(hpool.losses * hyp_scores_softmax)

        # Total loss = Exp Loss + Entropy Loss + Hypothesis Loss (weighted average).
        total_loss = self.est_beta * hpool.wavg.loss + \
            self.entropy_beta * softmax_entropy + \
            self.exp_beta * exp_loss

        # Update metrics.
        # TODO: Could reduce the number of lines.
        metrics.best.update(hpool.best.loss, hpool.best.pose)
        metrics.worst.update(hpool.worst.loss, hpool.worst.pose)
        metrics.most.update(hpool.most.loss, hpool.most.pose)
        metrics.least.update(hpool.least.loss, hpool.least.pose)
        metrics.stoch.update(hpool.random.loss, hpool.random.pose)
        metrics.random.update(hpool.random.loss, hpool.random.pose)
        metrics.avg.update(hpool.avg.loss, hpool.avg.pose)
        metrics.wavg.update(hpool.wavg.loss, hpool.wavg.pose)
        metrics.triang.update(hpool.triang.loss, hpool.triang.pose)

        return total_loss, metrics, hpool



class PoseDSAC(DSAC):
    ''' Differentiable RANSAC for pose triangulation.
    
    '''

    def __init__(
        self, 
        hyps: int, 
        num_joints: int, 
        entropy_beta: float, 
        entropy_to_scores: bool,
        temp: float, 
        gumbel: bool, 
        hard: bool, 
        body_lengths_mode: bool, 
        weighted_selection: bool, 
        exp_beta: float, 
        est_beta: float,
        score_nn: Callable, 
        loss_function: LossFunction, 
        device: str = 'cpu'
    ) -> None:
        ''' PoseDSAC constructor.
        
            Parameters
            ----------
            hyps -- number of hypotheses (trials) for each PoseDSAC iteration
            loss_function -- function to estimate the quality of triangulated poses
            scale --- scalar, GT scale, used to obtain proper translation
            device --- 'cuda' or 'cpu'
        '''
        self.hyps = hyps
        self.num_joints = num_joints

        self.entropy_beta = entropy_beta
        self.temp = temp
        self.est_beta = est_beta
        self.exp_beta = exp_beta
        
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

    def __sample_hyp(self, est_2d_pose, Ks, Rs, ts, calc_triang):
        ''' Select a random subset of point correspondences and calculate R and t.

            Parameters
            ----------
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
            np.arange(len(all_view_combinations)), size=num_joints)

        # For each joint, use the selected view subsets to triangulate points.
        pose_3d = torch.zeros([num_joints, 3], dtype=torch.float32, device=self.device)
        baseline_pose = torch.zeros([num_joints, 3], dtype=torch.float32, device=self.device) \
            if calc_triang else None

        for joint_idx in range(num_joints):
            cidxs = all_view_combinations[selected_combination_idxs[joint_idx]]
            all_views_cidxs = all_view_combinations[-1]

            pose_3d[joint_idx] = self.__triangulate_joint(
                est_2d_pose, joint_idx, Ks, Rs, ts, cidxs
            )
            # Do not calculate baseline more than once for efficiency.
            if calc_triang:
                baseline_pose[joint_idx] = self.__triangulate_joint(
                    est_2d_pose, joint_idx, Ks, Rs, ts, all_views_cidxs
                )

        return pose_3d, baseline_pose

    def __score_nn(self, est_3d_pose, mean, std):
        ''' Feed 3D pose coordinates into ScoreNN to obtain score for the hyp.

            Parameters
            ----------
            est_3d_pose
        '''
        # Standardize pose.
        est_3d_pose_norm = ((est_3d_pose - mean) / std)

        # Zero-center around hip joint.
        est_3d_pose_norm = center_pelvis(est_3d_pose_norm)

        # Extract body part lengths.
        if self.body_lengths_mode == 1 or self.body_lengths_mode == 2:
            connections = CONNECTIVITY_DICT['human36m']
            lengths = []
            for (kpt1, kpt2) in connections:
                lengths.append(torch.norm(est_3d_pose_norm[kpt1] - est_3d_pose_norm[kpt2]))
            lengths = torch.stack(lengths, dim=0)

        # Orient pose towards the positive Z-axis.
        n0 = torch.tensor([0., 0., 1.], dtype=torch.float32)
        THREE_KPTS = [KPTS['lshoulder'], KPTS['rshoulder'], KPTS['pelvis']]
        (a, b, c) = est_3d_pose_norm[THREE_KPTS[0]], \
            est_3d_pose_norm[THREE_KPTS[1]], \
            est_3d_pose_norm[THREE_KPTS[2]]
        n1 = torch.cross(c - a, b - a)

        axis = torch.cross(n0, n1) / torch.norm(torch.cross(n0, n1))
        angle = torch.acos(torch.dot(n0, n1))

        Rmat = angle_axis_to_rotation_matrix((angle * axis).unsqueeze(dim=0))
        est_3d_pose_norm = est_3d_pose_norm @ Rmat[0]

        # Select network input based on body lengths mode.
        if self.body_lengths_mode == 0:
            network_input = est_3d_pose_norm.flatten()
        elif self.body_lengths_mode == 1:
            network_input = torch.cat((est_3d_pose_norm.flatten(), lengths), dim=0)
        elif self.body_lengths_mode == 2:
            network_input = lengths

        return self.score_nn(network_input.cuda())

    def __call__(self, est_2d_pose, Ks, Rs, ts, gt_3d, mean, std, metrics):
        ''' Perform robust, differentiable triangulation.

            Parameters
            ----------
            est_2d_pose -- predicted 2D points for B frames: (CxJx2)
            Ks -- GT/estimated intrinsics for B frames: (Cx3x3)
            Rs -- GT/estimated rotations for B frames: (Cx3x3)
            ts -- GT/estimated translations for B frames: (Cx3x1)
            gts_3d -- GT 3D poses: (Jx3)
        '''
        hpool = HypothesisPool(self.hyps, self.num_joints, gt_3d, self.loss_function, device=self.device)

        for _ in range(0, self.hyps):
            sample_tuple = self.__sample_hyp(est_2d_pose, Ks, Rs, ts, hpool.triang is None)
            score = self.__score_nn(sample_tuple[0], mean, std)

            #score = torch.nan_to_num(score)
            #score[score != score] = 0.
            #score = torch.where(torch.isnan(score), torch.zeros_like(score), score)

            if hpool.triang is None:
                hpool.set_triang(sample_tuple[1])
            hpool.append(sample_tuple[0], score)

            if torch.isnan(score):
                print('')

            if torch.isnan(sample_tuple[0][0][0]):
                print('')

        # Softmax distribution from hypotheses scores.
        if self.gumbel:
            hyp_scores_softmax = F.gumbel_softmax(hpool.scores, tau=self.temp, hard=self.hard, dim=0)
        else:  
            hyp_scores_softmax = F.softmax(hpool.scores / self.temp, dim=0)

        # Entropy loss.
        if self.entropy_to_scores:
            #softmax_entropy = -torch.sum(hyp_scores * torch.log(hyp_scores))
            softmax_entropy = -torch.sum(hpool.scores * torch.log(hyp_scores_softmax))
        else:
            softmax_entropy = -torch.sum(hyp_scores_softmax * torch.log(hyp_scores_softmax))

        # Expectation loss.
        hpool.losses /= hpool.losses.max()
        exp_loss = torch.sum(hpool.losses * hyp_scores_softmax)

        # Total loss = Exp Loss + Entropy Loss + Hypothesis Loss (weighted average).
        total_loss = self.est_beta * hpool.wavg.loss + \
            self.entropy_beta * softmax_entropy + \
            self.exp_beta * exp_loss

        # Update metrics.
        # TODO: Could reduce the number of lines.
        metrics.best.update(hpool.best.loss, hpool.best.pose)
        metrics.worst.update(hpool.worst.loss, hpool.worst.pose)
        metrics.most.update(hpool.most.loss, hpool.most.pose)
        metrics.least.update(hpool.least.loss, hpool.least.pose)
        metrics.stoch.update(hpool.random.loss, hpool.random.pose)
        metrics.random.update(hpool.random.loss, hpool.random.pose)
        metrics.avg.update(hpool.avg.loss, hpool.avg.pose)
        metrics.wavg.update(hpool.wavg.loss, hpool.wavg.pose)
        metrics.triang.update(hpool.triang.loss, hpool.triang.pose)

        return total_loss, metrics, hpool
