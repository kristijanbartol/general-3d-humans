import torch
from collections import OrderedDict


class Hypothesis():

    def __init__(self, est_pose, gt_pose, loss_fun):
        self.est_pose = est_pose
        self.gt_pose = gt_pose
        self.loss_fun = loss_fun

        self.score = None

    @property
    def loss(self):
        return self.loss_fun(self.est_pose, self.gt_pose)

    @property
    def pose(self):
        return self.est_pose


class HypothesisPool():

    def __init__(self, nhyps, njoints, gt_3d, loss_fun, device):
        self.nhyps = nhyps
        self.njoints = njoints

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
    def _top_idx(self):
        return self.sorted_score_idxs[0].item()

    @property
    def top(self) -> Hypothesis:
        # Best per score.
        return self.__create_hyp(self._top_idx)

    @property
    def _bottom_idx(self):
        return self.sorted_score_idxs[-1].item()

    @property
    def bottom(self) -> Hypothesis:
        # Worst per score.
        return self.__create_hyp(self._bottom_idx)

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
        return Hypothesis(pose, self.gt_3d, self.loss_fun)

    @property
    def gt(self):
        return Hypothesis(self.gt_3d, self.gt_3d, self.loss_fun)

    def set_triang(self, pose) -> None:
        self.triang = Hypothesis(pose, self.gt_3d, self.loss_fun)

    def get_qualitative_hyps_dict(self):
        hyps_dict = OrderedDict()
        hyps_dict['wavg'] = self.wavg.pose.detach().numpy()
        hyps_dict['top'] = self.top.pose.detach().numpy()
        hyps_dict['bottom'] = self.bottom.pose.detach().numpy()
        hyps_dict['random'] = self.random.pose.detach().numpy()
        hyps_dict['triang'] = self.triang.pose.detach().numpy()
        hyps_dict['gt'] = self.gt.pose.detach().numpy()

        return hyps_dict
