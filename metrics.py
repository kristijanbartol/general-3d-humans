import torch
import numpy as np


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
            lengths[i] = np.linalg.norm(pose_3d[i][0] - pose_3d[i][1], ord=2)
        ratios = np.empty([self.nseg, self.nseg], dtype=np.float32)
        for i in range(self.nseg):
            for j in range(self.nseg):
                ratios[i, j] = lengths[i] / lengths[j]

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
        self.worst = PoseMetrics()
        self.top = PoseMetrics()
        self.bottom = PoseMetrics()
        #self.gt = PoseMetrics()
        self.random = PoseMetrics()
        self.avg = PoseMetrics()
        self.wavg = PoseMetrics()
        self.triang = PoseMetrics()

        self.loss = LossMetrics()
