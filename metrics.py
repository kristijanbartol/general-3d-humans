from collections import OrderedDict
import torch
import numpy as np

from mvn.utils.vis import CONNECTIVITY_DICT, SEGMENT_IDXS


def mpjpe(est, gt):
    return torch.mean(torch.norm(est - gt, p=2, dim=1))


def center_pelvis(pose_3d):
    if pose_3d.shape[0] == 17:
        return pose_3d - pose_3d[6, :]
    else:
        return pose_3d - pose_3d[2, :]


def rel_mpjpe(est, gt):
    est_centered = center_pelvis(est)
    gt_centered = center_pelvis(gt)

    return mpjpe(est_centered, gt_centered)


class RatioVariances():

    def __init__(self, ratioss, dataset):
        segment_idxs = SEGMENT_IDXS[dataset]

        self.upper_arm = ratioss[:, segment_idxs[3], segment_idxs[6]].var()
        self.lower_arm = ratioss[:, segment_idxs[4], segment_idxs[7]].var()
        self.shoulder  = ratioss[:, segment_idxs[2], segment_idxs[5]].var()
        self.hip       = ratioss[:, segment_idxs[9], segment_idxs[12]].var()
        self.upper_leg = ratioss[:, segment_idxs[10], segment_idxs[13]].var()
        self.lower_leg = ratioss[:, segment_idxs[11], segment_idxs[14]].var()

        self.right_left = \
            (self.upper_arm + \
            self.lower_arm + \
            self.shoulder + \
            self.hip + \
            self.upper_leg + \
            self.lower_leg) / 6.
        #self.all = ratioss[:, 9, 12].var()


class PoseMetrics():

    def __init__(self, dataset):
        #self._all_segments = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
        #[5, 6], [0, 7], [7, 8], [8, 14], [14, 15], [15, 16],
        #[8, 11], [11, 12], [12, 13], [8, 9], [9, 10]]

        self.dataset = dataset
        self.segments = CONNECTIVITY_DICT[self.dataset]
        self.nseg = len(self.segments)

        self.ratioss = None
        self._errors = None
        self.flush()
    
    def update(self, error, pose_3d):
        self._errors.append(error.detach().numpy())

        pose_3d = pose_3d.detach().numpy()
        lengths = np.empty(self.nseg, dtype=np.float32)
        for i, seg_idxs in enumerate(self.segments):
            lengths[i] = np.linalg.norm(pose_3d[seg_idxs[0]] - pose_3d[seg_idxs[1]], ord=2)
        ratios = np.empty([1, self.nseg, self.nseg], dtype=np.float32)
        for i in range(self.nseg):
            for j in range(self.nseg):
                ratios[0, i, j] = lengths[i] / lengths[j]

        self.ratioss = np.concatenate([self.ratioss, ratios], axis=0)

    @property
    def ratio_variances(self):
        return RatioVariances(self.ratioss, self.dataset)

    @property
    def recent_error(self):
        nlast = min(100, len(self._errors))
        return np.array(self._errors[-nlast:]).mean()

    @property
    def error(self):
        return np.array(self._errors).mean()

    @property
    def errors(self):
        return np.array(self._errors)

    def flush(self):
        self.ratioss = np.empty([0, self.nseg, self.nseg], dtype=np.float32)
        self._errors = []


class GlobalMetrics():

    def __init__(self, dataset):
        self.best = PoseMetrics(dataset)
        self.worst = PoseMetrics(dataset)
        self.top = PoseMetrics(dataset)
        self.bottom = PoseMetrics(dataset)
        self.random = PoseMetrics(dataset)
        self.avg = PoseMetrics(dataset)
        self.wavg = PoseMetrics(dataset)
        self.triang = PoseMetrics(dataset)

    @property
    def diff_to_triang(self):
        return (self.wavg.errors - self.triang.errors).mean()

    @property
    def diff_to_avg(self):
        return (self.wavg.errors - self.avg.errors).mean()

    @property
    def diff_to_random(self):
        return (self.wavg.errors - self.random.errors).mean()

    def flush(self):
        for attribute in list(self.__dict__):
            self.__dict__[attribute].flush()

    # TODO: Simplify these.
    def get_overall_metrics_dict(self):
        metrics_dict = OrderedDict()

        metrics_dict['best'] = [self.best.error]
        metrics_dict['wavg'] = [self.wavg.error]
        metrics_dict['avg'] = [self.avg.error]
        metrics_dict['top'] = [self.top.error]
        metrics_dict['random'] = [self.random.error]
        metrics_dict['bottom'] = [self.bottom.error]
        metrics_dict['worst'] = [self.worst.error]
        metrics_dict['triang'] = [self.triang.error]
        metrics_dict['ransac'] = [27.4]

        return metrics_dict

    def get_pose_prior_metrics_dict(self):
        metrics_dict = OrderedDict()

        metrics_dict['best'] = [self.best.ratio_variances.right_left]
        metrics_dict['wavg'] = [self.wavg.ratio_variances.right_left]
        metrics_dict['avg'] = [self.avg.ratio_variances.right_left]
        metrics_dict['top'] = [self.top.ratio_variances.right_left]
        metrics_dict['random'] = [self.random.ratio_variances.right_left]
        metrics_dict['bottom'] = [self.bottom.ratio_variances.right_left]
        metrics_dict['worst'] = [self.worst.ratio_variances.right_left]
        metrics_dict['triang'] = [self.triang.ratio_variances.right_left]

        return metrics_dict
