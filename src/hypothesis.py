import torch
import numpy as np
import kornia
from collections import OrderedDict

# TODO: Create a single abstract class for pose and camera hypotheses.


class Hypothesis():

    def __init__(self, est_pose, gt_pose=None, loss_fun=None):
        self.est_pose = est_pose
        self.gt_pose = gt_pose
        self.loss_fun = loss_fun

        self.score = None

    @property
    def loss(self):
        if self.loss_fun is None:
            return 0.
        return self.loss_fun(self.est_pose, self.gt_pose)

    @property
    def pose(self):
        return self.est_pose


class HypothesisPool():

    def __init__(self, nhyps, njoints, gt_3d=None, loss_fun=None, device='cpu'):
        self.nhyps = nhyps
        self.njoints = njoints

        # TODO: Propagate the case when loss function is None (for inference).
        self.gt_3d = gt_3d
        self.loss_fun = loss_fun
        self.device = device

        self.hyps = []

        # These attributes are used for internal simplicity, 
        # should correspond to self.hyps.
        self.losses = torch.zeros([nhyps, 1], device=self.device)
        self.scores = torch.zeros([nhyps, 1], device=self.device)
        self.poses = torch.zeros([nhyps, self.njoints, 3], device=self.device)

        # TODO: 4-triang is currently set externally.
        self.triang = None

    def append(self, pose, score):
        idx = len(self.hyps) - 1
        hyp = Hypothesis(pose, self.gt_3d, self.loss_fun)
        self.losses[idx] = hyp.loss
        self.scores[idx] = score
        hyp.score = score
        self.poses[idx] = pose
        self.hyps.append(hyp)

    @property
    def sorted_losses(self):
        return torch.sort(self.losses, dim=0)[0]

    @property
    def sorted_loss_idxs(self):
        return torch.sort(self.losses, dim=0)[1]

    @property
    def sorted_scores(self):
        return torch.sort(self.scores, dim=0, descending=True)[0]

    @property
    def sorted_score_idxs(self):
        return torch.sort(self.scores, dim=0, descending=True)[1]

    @property
    def losses_sorted_by_scores(self):
        return self.losses[self.sorted_score_idxs[:, 0]]

    @property
    def scores_sorted_by_losses(self):
        return self.scores[self.sorted_loss_idxs[:, 0]]

    def __create_hyp(self, idx):
        hyp = Hypothesis(self.poses[idx], self.gt_3d, self.loss_fun)
        hyp.score = self.scores[idx]
        return hyp

    @property
    def _best_idx(self):
        return self.sorted_loss_idxs[0].item()

    @property
    def best(self) -> Hypothesis:
        # Best per loss.
        return self.__create_hyp(self._best_idx)

    @property
    def _worst_idx(self):
        return self.sorted_loss_idxs[-1].item()

    @property
    def worst(self) -> Hypothesis:
        # Worst per loss.
        return self.__create_hyp(self._worst_idx)

    @property
    def _most_idx(self):
        return self.sorted_score_idxs[0].item()

    @property
    def most(self) -> Hypothesis:
        # Best per score.
        return self.__create_hyp(self._most_idx)

    @property
    def _least_idx(self):
        return self.sorted_score_idxs[-1].item()

    @property
    def least(self) -> Hypothesis:
        # Worst per score.
        return self.__create_hyp(self._least_idx)

    @property
    def stoch(self) -> Hypothesis:
        # Random hypothesis, based on estimated distribution.
        stoch_selection = torch.multinomial(self.scores[:, 0], num_samples=1)[0]
        return self.__create_hyp(stoch_selection)

    @property
    def random(self) -> Hypothesis:
        return self.__create_hyp(torch.randint(self.nhyps, (1,))[0])

    @property
    def avg(self):
        pose = torch.zeros((self.njoints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            pose += self.poses[hidx] * 1.0
        pose /= self.nhyps
        return Hypothesis(pose, self.gt_3d, self.loss_fun)

    @property
    def wavg(self):
        pose = torch.zeros((self.njoints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            pose += self.poses[hidx] * self.scores[hidx]
        pose /= self.sorted_scores.sum()
        if torch.isnan(pose[0][0]):
            print('')
        return Hypothesis(pose, self.gt_3d, self.loss_fun)

    @property
    def gt(self):
        return Hypothesis(self.gt_3d, self.gt_3d, self.loss_fun)

    def set_triang(self, pose) -> None:
        self.triang = Hypothesis(pose, self.gt_3d, self.loss_fun)

    def get_qualitative_hyps_dict(self):
        hyps_dict = OrderedDict()
        hyps_dict['weight'] = self.wavg.pose.detach().numpy()
        hyps_dict['most'] = self.most.pose.detach().numpy()
        hyps_dict['least'] = self.least.pose.detach().numpy()
        hyps_dict['stoch'] = self.stoch.pose.detach().numpy()
        hyps_dict['naive'] = self.triang.pose.detach().numpy()
        hyps_dict['gt'] = self.gt.pose.detach().numpy()

        return hyps_dict

    
class CameraParams():

    def __init__(self, K1, K2, R1, t1, R_rel, t_rel):
        self.K1 = K1
        self.K2 = K2
        self.R1 = R1
        self.t1 = t1
        self.R_rel = R_rel
        self.t_rel = t_rel

    @property
    def Ks(self):
        return (self.K1, self.K2)

    @property
    def R2(self):
        return self.R1 * self.R_rel

    @property
    def t2(self):
        return self.t1 * self.t_rel

    @property
    def quat_R(self):
        return kornia.rotation_matrix_to_quaternion(self.R_rel)

    @staticmethod
    def to_R_rel(R1, R2):
        return R2 @ torch.inverse(R1)

    @staticmethod
    def to_t_rel(t1, t2, R1, R2):
        return -R2 @ torch.inverse(R1) @ t1 + t2


class CameraHypothesis():

    def __init__(self, est_params, gt_params, random_points, loss_fun):
        self.est_params = est_params
        self.gt_params = gt_params
        self.random_points = random_points
        self.loss_fun = loss_fun

        self.score = None

    @property
    def loss(self):
        return self.loss_fun(self.est_params, self.gt_params)

    @property
    def params(self):
        return self.est_params


class CameraHypothesisPool():

    def __init__(self, nhyps, gt_Ks, gt_Rs, gt_ts, random_points, loss_fun, device):
        self.nhyps = nhyps

        self.gt_Ks = gt_Ks
        self.gt_Rs = gt_Rs
        self.gt_ts = gt_ts
        self.random_points = random_points
        self.loss_fun = loss_fun
        self.device = device

        self.hyps = []

        # These attributes are used for internal simplicity, 
        # should correspond to self.hyps.
        self.losses = torch.zeros([nhyps, 1], device=self.device)
        self.scores = torch.zeros([nhyps, 1], device=self.device)
        self.params = []

    def __create_hyp(self, idx):
        hyp = CameraHypothesis(self.params[idx], self.gt_params, self.random_points, self.loss_fun)
        hyp.score = self.scores[idx]
        return hyp

    def append(self, params, score):
        idx = len(self.hyps) - 1
        hyp = CameraHypothesis(params, self.gt_params, self.random_points, self.loss_fun)
        self.losses[idx] = hyp.loss
        self.scores[idx] = score
        hyp.score = score
        self.params.append(params)
        self.hyps.append(hyp)

    @property
    def sorted_losses(self):
        return torch.sort(self.losses, dim=0)[0]

    @property
    def sorted_loss_idxs(self):
        return torch.sort(self.losses, dim=0)[1]

    @property
    def sorted_scores(self):
        return torch.sort(self.scores, dim=0, descending=True)[0]

    @property
    def sorted_score_idxs(self):
        return torch.sort(self.scores, dim=0, descending=True)[1]

    @property
    def losses_sorted_by_scores(self):
        return self.losses[self.sorted_score_idxs[:, 0]]

    @property
    def scores_sorted_by_losses(self):
        return self.scores[self.sorted_loss_idxs[:, 0]]

    @property
    def _best_idx(self):
        return self.sorted_loss_idxs[0].item()

    @property
    def best(self) -> CameraHypothesis:
        # Best per loss.
        return self.__create_hyp(self._best_idx)

    @property
    def _worst_idx(self):
        return self.sorted_loss_idxs[-1].item()

    @property
    def worst(self) -> CameraHypothesis:
        # Worst per loss.
        return self.__create_hyp(self._worst_idx)

    @property
    def _most_idx(self):
        return self.sorted_score_idxs[0].item()

    @property
    def most(self) -> CameraHypothesis:
        # Best per score.
        return self.__create_hyp(self._most_idx)

    @property
    def _least_idx(self):
        return self.sorted_score_idxs[-1].item()

    @property
    def least(self) -> CameraHypothesis:
        # Worst per score.
        return self.__create_hyp(self._least_idx)

    @property
    def stoch(self) -> CameraHypothesis:
        # Random hypothesis, based on estimated distribution.
        stoch_selection = torch.multinomial(self.scores[:, 0], num_samples=1)[0]
        return self.__create_hyp(stoch_selection)

    @property
    def random(self) -> CameraHypothesis:
        return self.__create_hyp(torch.randint(self.nhyps, (1,))[0])

    @property
    def avg(self):
        #params = torch.zeros((self.njoints, 3), dtype=torch.float32, device=self.device)
        quat_R = torch.zeros(4, dtype=torch.float32, device=self.device)
        t = torch.zeros((3, 1), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            quat_R += self.params[hidx].quat_R * 1.0
            t += self.params[hidx].t_rel * 1.0
        quat_R /= self.nhyps
        t /= self.nhyps
        R = kornia.quaternion_to_rotation_matrix(quat_R)
        params = CameraParams(self.Ks[0], self.Ks[1], self.Rs[0], self.ts[0], R, t)
        return CameraHypothesis(params, self.gt_params, self.loss_fun)

    @property
    def wavg(self):
        quat_R = torch.zeros(4, dtype=torch.float32, device=self.device)
        t = torch.zeros((3, 1), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            quat_R += self.params[hidx].quat_R * self.scores[hidx]
            t += self.params[hidx].t_rel * self.scores[hidx]
        quat_R /= self.sorted_scores.sum()
        t /= self.sorted_scores.sum()
        R = kornia.quaternion_to_rotation_matrix(quat_R)
        params = CameraParams(self.Ks[0], self.Ks[1], self.Rs[0], self.ts[0], R, t)
        return CameraHypothesis(params, self.gt_params, self.loss_fun)

    @property
    def gt(self):
        return CameraHypothesis(self.gt_params, self.gt_params, self.random_points, self.loss_fun)
