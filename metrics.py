import torch


def mpjpe(est, gt):
    return torch.mean(torch.norm(est - gt, p=2, dim=1))


def center_pelvis(pose_3d):
    return pose_3d - pose_3d[6, :]


def rel_mpjpe(est, gt):
    est_centered = center_pelvis(est)
    gt_centered = center_pelvis(gt)

    return mpjpe(est_centered, gt_centered)


class RatioVariances():

    def __init__(self, ratioss):
        self.upper_arm = ratioss[:, 9, 12].var()
        self.lower_arm = ratioss[:, 10, 13].var()
        self.shoulder  = ratioss[:, 8, 11].var()
        self.hip       = ratioss[:, 0, 3].var()
        self.upper_leg = ratioss[:, 1, 4].var()
        self.lower_leg = ratioss[:, 2, 5].var()

        self.right_left = \
            (self.upper_arm + \
            self.lower_arm + \
            self.shoulder + \
            self.hip + \
            self.upper_leg + \
            self.lower_leg) / 6.
        #self.all = ratioss[:, 9, 12].var()


class PoseMetrics():

    def __init__(self):
        self._all_segments = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
        [5, 6], [0, 7], [7, 8], [8, 14], [14, 15], [15, 16],
        [8, 11], [11, 12], [12, 13], [8, 9], [9, 10]]

        self.nseg = len(self._all_segments)

        self.ratioss = None
        self.errors = None
        self.flush()
    
    def update(self, error, pose_3d):
        self.errors.append(error)

        lengths = np.empty(self.nseg, dtype=np.float32)
        for i in range(self.nseg):
            lengths[i] = np.linalg.norm(pose_3d[i][0] - pose_3d[i][1], ord=2))
        ratios = np.empty([self.nseg, self.nseg], dtype=np.float32)
        for i in range(self.nseg):
            for j in range(self.nseg):
                ratios[i, j] = length[i] / length[j]

        self.ratioss = np.concatenate([self.ratioss, ratios], axis=0)

    @property
    def ratio_variances(self):
        return RatioVariances(self.ratioss)

    @property
    def recent_error(self):
        nlast = min(100, len(self.errors))
        return np.array(self.errors[-nlast:]).mean()

    @property
    def error(self):
        return np.array(self.errors).mean()

    def flush(self):
        self.ratioss = np.empty([0, self.nseg, self.nseg], dtype=np.float32)
        self.errors = []


class LossMetrics():

    def __init__(self):
        self._total = None
        self._expect = None
        self._entropy = None
        self.flush()

    def update(self, total, expect, entropy):
        self._total.append(total)
        self._expect.append(expect)
        self._entropy.append(entropy)

    @property
    def total(self):
        nlast = min(100, len(self._total))
        return np.array(self._total[-nlast:]).mean()

    @property
    def expect(self):
        nlast = min(100, len(self._expect))
        return np.array(self._expect[-nlast:]).mean()

    @property
    def entropy(self):
        nlast = min(100, len(self._entropy))
        return np.array(self._entropy[-nlast:]).mean()

    def flush(self):
        self._total = []
        self._expect = []
        self._entropy = []


class GlobalMetrics():

    def __init__(self):
        self.best = PoseMetrics()
        self.gt = PoseMetrics()
        self.random = PoseMetrics()
        self.avg = PoseMetrics()
        self.wavg = PoseMetrics()
        self.triang = PoseMetrics()
        self.selected = PoseMetrics()
        self.worst = PoseMetrics()

        self.loss = LossMetrics()


class Hypotheses():

    def __init__(self, nhyps, njoints, device=device):
        self.nhyps = nhyps
        self.njoints = njoints
        self.hyp_losses = torch.zeros([nhyps, 1], device=device)
        self.hyp_scores = torch.zeros([nhyps, 1], device=device)
        self.hyp_poses = torch.zeros([nhyps, self.num_joints, 3], device=device)

    def update(self, idx, loss, score, pose):
        self.hyp_losses[idx] = loss
        self.hyp_scores[idx] = score
        self.hyp_poses[idx] = pose

    @property
    def sorted_losses(self):
        return torch.sort(self.hyp_losses, dim=0)[0]

    @property
    def sorted_loss_idxs(self):
        return torch.sort(hyp_losses, dim=0)[1]

    @property
    def sorted_scores(self):
        return torch.sort(self.hyp_scores, dim=0, descending=True)[0]

    @property
    def sorted_score_idxs(self):
        return torch.sort(self.hyp_scores, dim=0, descending=True)[1]

    @property
    def losses_sorted_by_scores(self):
        return hyp_losses[self.sorted_score_idxs[:, 0]]

    @property
    def scores_sorted_by_losses(self):
        return hyp_scores[self.sorted_loss_idxs[:, 0]]

    @property
    def best(self):
        # Best per loss.
        return self.hyp_poses[self.sorted_loss_idxs[0]]

    @property
    def worst(self):
        # Worst per loss.
        return self.hyp_poses[self.sorted_loss_idxs[-1]]

    @property
    def top(self):
        # Best per score.
        return self.hyp_poses[self.sorted_score_idxs[0]]

    @property
    def bottom(self):
        # Worst per score.
        return self.hyp_poses[self.sorted_score_idxs[-1]]

    @property
    def random(self):
        return self.hyp_poses[torch.randint(self.nhyps, (1,))[0]]

    @property
    def avg(self):
        avg_pose = torch.zeros((self.njoints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            avg_pose += self.hyp_poses[self.sorted_score_idxs[:, 0]][hidx] * 1.0
        avg_pose /= self.nhyps
        return avg_pose

    @property
    def wavg(self):
        weight_avg_pose = torch.zeros((self.njoints, 3), dtype=torch.float32, device=self.device)
        for hidx in range(self.nhyps):
            weight_avg_pose += self.hyp_poses[self.sorted_score_idxs[:, 0]][hidx] * self.sorted_scores[hidx, 0]
        weight_avg_pose /= self.nhyps
        return weight_avg_pose
