import torch


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

        self.hyps = []

        # These attributes are used for internal simplicity, 
        # should correspond to self.hyps.
        self.losses = torch.zeros([nhyps, 1], device=device)
        self.scores = torch.zeros([nhyps, 1], device=device)
        self.poses = torch.zeros([nhyps, self.num_joints, 3], device=device)

        # TODO: Baseline is currently set externally.
        self.baseline = None

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

    @property
    def best(self):
        # Best per loss.
        return self.poses[self.sorted_loss_idxs[0]]

    @property
    def worst(self):
        # Worst per loss.
        return self.poses[self.sorted_loss_idxs[-1]]

    @property
    def top(self):
        # Best per score.
        return self.poses[self.sorted_score_idxs[0]]

    @property
    def bottom(self):
        # Worst per score.
        return self.poses[self.sorted_score_idxs[-1]]

    @property
    def random(self):
        return self.poses[torch.randint(self.nhyps, (1,))[0]]

    @property
    def avg(self):
        avg_pose = torch.zeros((self.njoints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            avg_pose += self.poses[self.sorted_score_idxs[:, 0]][hidx] * 1.0
        avg_pose /= self.nhyps
        return avg_pose

    @property
    def wavg(self):
        weight_avg_pose = torch.zeros((self.njoints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            weight_avg_pose += self.poses[self.sorted_score_idxs[:, 0]][hidx] * self.sorted_scores[hidx, 0]
        weight_avg_pose /= self.nhyps
        return weight_avg_pose

    def set_baseline(self, pose):
        self.baseline = Hypothesis(pose, self.gt_3d, self.loss_fun)
