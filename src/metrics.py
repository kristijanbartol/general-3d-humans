# Author: Kristijan Bartol


from collections import OrderedDict
import torch
import numpy as np
import sys

sys.path.append('/general-3d-humans/')
from src.const import CONNECTIVITY_DICT, SEGMENT_IDXS


def mpjpe(est: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    '''MPJPE between the estimated and the target (GT) 3D pose.
    
        MPJPE is calculated as the mean Euclidean distance
        between the corresponding points, originally defined in:        
        "Human3.6M: Large Scale Datasets and Predictive Methods 
        for 3D Human Sensing in Natural Environments"
    
        Parameters
        ----------
        :param est: estimated 3D pose (bsize, J, 3)
        :param gt: ground-truth (target) 3D pose (bsize, J, 3)
        :return: Euclidean distance between corresponding joints
    '''
    return torch.mean(torch.norm(est - gt, p=2, dim=1))


def center_pelvis(pose_3d: torch.Tensor) -> torch.Tensor:
    '''Move the 3D pose to the origin (pelvis-wise).
    
        In case of H36M joint set, the pelvis is at index 6.
        In case of CMU joint set, the pelvis is at index 2.
        
        Parameters
        ----------
        :param pose_3d: 3D pose (bsize, J, 3)
        :return: centered 3D pose
    '''
    if pose_3d.shape[0] == 17:
        return pose_3d - pose_3d[6, :]
    else:
        return pose_3d - pose_3d[2, :]


def rel_mpjpe(est: torch.Tensor, 
              gt: torch.Tensor, 
              device: str = 'cpu'
    ) -> torch.Tensor:
    '''Relative MPJPE.
    
        Calculated by first translating estimation and target
        to origin and calculating MPJPE score
        
        Parameters
        ----------
        :param est: estimated 3D pose (bsize, J, 3)
        :param gt: ground-truth (target) 3D pose (bsize, J, 3)
        :param device: cuda or cpu
        :return: MPJPE score between the prediction and target (bsize,)
    '''
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

        self.left_right = \
            (self.upper_arm + \
            self.lower_arm + \
            self.shoulder + \
            self.hip + \
            self.upper_leg + \
            self.lower_leg) / 6.
        #self.all = ratioss[:, 9, 12].var()


class PoseMetrics():

    def __init__(self, dataset):

        self.dataset = dataset
        self.segments = CONNECTIVITY_DICT[self.dataset]
        self.nseg = len(self.segments)

        self.ratioss = None
        self._errors = None
        self.flush()
    
    def update(self, error, pose_3d):
        if not np.isnan(error.detach().numpy()):
            self._errors.append(error.detach().numpy())
        else:
            self._errors.append(self._errors[-1])
            print('Skipping NaN!')
            return

        pose_3d = pose_3d.detach().numpy()
        lengths = np.empty(self.nseg, dtype=np.float32)
        for i, seg_idxs in enumerate(self.segments):
            if self.dataset == 'cmu':
                lengths[i] = 0.
            else:
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
        self.most = PoseMetrics(dataset)
        self.least = PoseMetrics(dataset)
        self.stoch = PoseMetrics(dataset)
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

        metrics_dict['weight'] = [self.wavg.error]
        metrics_dict['avg'] = [self.avg.error]
        metrics_dict['most'] = [self.most.error]
        metrics_dict['least'] = [self.least.error]
        metrics_dict['stoch'] = [self.stoch.error]
        metrics_dict['random'] = [self.random.error]
        metrics_dict['best'] = [self.best.error]
        metrics_dict['worst'] = [self.worst.error]
        metrics_dict['naive'] = [self.triang.error]
        metrics_dict['ransac'] = [27.4]

        return metrics_dict

    def get_pose_prior_metrics_dict(self):
        metrics_dict = OrderedDict()

        values = [[self.wavg.ratio_variances.upper_arm, self.avg.ratio_variances.upper_arm, self.most.ratio_variances.upper_arm,
            self.least.ratio_variances.upper_arm, self.stoch.ratio_variances.upper_arm, self.random.ratio_variances.upper_arm,
            self.best.ratio_variances.upper_arm, self.worst.ratio_variances.upper_arm],

            [self.wavg.ratio_variances.lower_arm, self.avg.ratio_variances.lower_arm, self.most.ratio_variances.lower_arm,
            self.least.ratio_variances.lower_arm, self.stoch.ratio_variances.lower_arm, self.random.ratio_variances.lower_arm,
            self.best.ratio_variances.lower_arm, self.worst.ratio_variances.lower_arm],

            [self.wavg.ratio_variances.shoulder, self.avg.ratio_variances.shoulder, self.most.ratio_variances.shoulder,
            self.least.ratio_variances.shoulder, self.stoch.ratio_variances.shoulder, self.random.ratio_variances.shoulder,
            self.best.ratio_variances.shoulder, self.worst.ratio_variances.shoulder],

            [self.wavg.ratio_variances.hip, self.avg.ratio_variances.hip, self.most.ratio_variances.hip,
            self.least.ratio_variances.hip, self.stoch.ratio_variances.hip, self.random.ratio_variances.hip,
            self.best.ratio_variances.hip, self.worst.ratio_variances.hip],

            [self.wavg.ratio_variances.upper_leg, self.avg.ratio_variances.upper_leg, self.most.ratio_variances.upper_leg,
            self.least.ratio_variances.upper_leg, self.stoch.ratio_variances.upper_leg, self.random.ratio_variances.upper_leg,
            self.best.ratio_variances.upper_leg, self.worst.ratio_variances.upper_leg],

            [self.wavg.ratio_variances.lower_leg, self.avg.ratio_variances.lower_leg, self.most.ratio_variances.lower_leg,
            self.least.ratio_variances.lower_leg, self.stoch.ratio_variances.lower_leg, self.random.ratio_variances.lower_leg,
            self.best.ratio_variances.lower_leg, self.worst.ratio_variances.lower_leg]
        ]

        return values



class CameraMetrics():

    def __init__(self):
        self._errors = None
        self.flush()
    
    def update(self, error):
        self._errors.append(error.detach().numpy())

    @property
    def error(self):
        return np.array(self._errors).mean()

    @property
    def errors(self):
        return np.array(self._errors)

    def flush(self):
        self._errors = []



class CameraGlobalMetrics():

    def __init__(self):
        self.best = CameraMetrics()
        self.worst = CameraMetrics()
        self.most = CameraMetrics()
        self.least = CameraMetrics()
        self.stoch = CameraMetrics()
        self.random = CameraMetrics()
        self.avg = CameraMetrics()
        self.wavg = CameraMetrics()
        self.triang = CameraMetrics()

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

        metrics_dict['weight'] = [self.wavg.error]
        metrics_dict['avg'] = [self.avg.error]
        metrics_dict['most'] = [self.most.error]
        metrics_dict['least'] = [self.least.error]
        metrics_dict['stoch'] = [self.stoch.error]
        metrics_dict['random'] = [self.random.error]
        metrics_dict['best'] = [self.best.error]
        metrics_dict['worst'] = [self.worst.error]

        return metrics_dict
