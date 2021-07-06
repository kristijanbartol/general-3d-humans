import numpy as np
import torch
import os
import math
import random

from torch.utils.data import Dataset


TRAIN_SIDXS = [1, 5, 6, 7, 8]
TEST_SIDXS = [9, 11]
SUBJECT_ORD_MAP = {
        1:  0,
        5:  1,
        6:  2,
        7:  3,
        8:  4,
        9:  5,
        11: 6
    }

ORD_SUBJECT_MAP = {
    0: 1,
    1: 5,
    2: 6,
    3: 7,
    4: 8,
    5: 9,
    6: 10
}


class SparseDataset(Dataset):
    '''Temporary dataset class for Human3.6M dataset.'''

    def __init__(self, rootdir, cam_idxs, num_frames=30, num_iterations=50000):
        '''The constructor loads and prepares predictions, GTs, and class parameters.

        rootdir --- the directory where the predictions, camera params and GT are located
        cam_idxs --- the subset of camera indexes used (the first is used as a reference)
        num_frames --- number of subset frames, M, used (which means P=M*J)
        num_iterations --- number of iterations per epoch, i.e. length of the dataset
        '''
        # Initialize data for each subject.
        self.Ks = dict.fromkeys(TRAIN_SIDXS + TEST_SIDXS)
        self.Rs = dict.fromkeys(TRAIN_SIDXS + TEST_SIDXS)
        self.ts = dict.fromkeys(TRAIN_SIDXS + TEST_SIDXS)
        self.preds_2d = dict.fromkeys(TRAIN_SIDXS + TEST_SIDXS)
        self.gt_3d = dict.fromkeys(TRAIN_SIDXS + TEST_SIDXS)
        self.bboxes = dict.fromkeys(TRAIN_SIDXS + TEST_SIDXS)

        self.cam_idxs = cam_idxs
        self.num_iterations = num_iterations
        self.num_frames = num_frames

        # Collect precalculated correspondences, camera params and and 3D GT,
        # for every subject, based on folder indexes.
        for dirname in os.listdir(rootdir):
            sidx = int(dirname[1:])
            Ks, Rs, ts = self.__load_camera_params(sidx, cam_idxs)
            self.Ks[sidx] = Ks
            self.Rs[sidx] = Rs
            self.ts[sidx] = ts

            pred_path = os.path.join(rootdir, dirname, 'all_2d_preds.npy')
            gt_path = os.path.join(rootdir, dirname, 'all_3d_gt.npy')
            bbox_path = os.path.join(rootdir, dirname, 'all_bboxes.npy')

            self.preds_2d[sidx] = np.load(pred_path)[:, cam_idxs]
            self.gt_3d[sidx] = np.load(gt_path)
            self.bboxes[sidx] = np.load(bbox_path)[:, cam_idxs]

            # Unbbox keypoints.
            bbox_height = np.abs(self.bboxes[sidx][:, :, 0, 0] - self.bboxes[sidx][:, :, 1, 0])
            self.preds_2d[sidx] *= np.expand_dims(
                np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
            self.preds_2d[sidx] += np.expand_dims(self.bboxes[sidx][:, :, 0, :], axis=2)

            # TODO: Obtain GT scale to estimate translation also.

    @staticmethod
    def __load_camera_params(subject_idx, cam_idxs):
        '''Loading camera parameters for given subject and camera subset.

        Loads camera parameters for a given subject and subset cameras.
        subject_idx --- subject index
        cam_idxs --- subset of camera indexes
        '''
        labels = np.load('/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', 
            allow_pickle=True).item()
        camera_params = labels['cameras'][SUBJECT_ORD_MAP[subject_idx]]

        Ks, Rs, ts = [], [], []
        for cam_idx in cam_idxs:
            Ks.append(camera_params[cam_idx][2])
            Rs.append(camera_params[cam_idx][0])
            ts.append(camera_params[cam_idx][1])

        Ks = np.stack(Ks, axis=0)
        Rs = np.stack(Rs, axis=0)
        ts = np.stack(ts, axis=0)

        return Ks, Rs, ts

    def __len__(self):
        return self.num_iterations

    # NOTE: Not using idx, might not DataLoader.
    def __getitem__(self, idx):
        '''
        Get random subset of point correspondences from the preset number of frames.

        Return:
        -------
        batch_point_corresponds -- [(C-1)xPx2x2]
        selected_gt_3d -- [FxJx3]
        batch_Ks -- [Cx2x3x3]
        batch_Rs -- [Cx2x3x3]
        batch_ts -- [Cx2x3x1]
        '''
        rand_sidx = ORD_SUBJECT_MAP[random.randint(0, len(TRAIN_SIDXS))]
        # TODO: At the moment, only have S9 keypoints.
        rand_sidx = 9

        # Selecting a subset of frames.
        selected_frames = np.random.choice(
            np.arange(self.preds_2d[rand_sidx].shape[0]), size=self.num_frames)

        # Select 2D predictions, 3D GT, and camera parameters 
        # for a given random subject and selected frames.
        selected_preds = self.preds_2d[rand_sidx][selected_frames]
        selected_gt_3d = self.gt_3d[rand_sidx][selected_frames]
        Ks = self.Ks[rand_sidx]
        Rs = self.Rs[rand_sidx]
        ts = self.ts[rand_sidx]

        # All points stacked along a single dimension for a single subject.
        point_corresponds = np.concatenate(
            np.split(selected_preds, selected_preds.shape[0], axis=0), axis=2)[0]
        # TODO: This can be simplified by removing dimension.
        #selected_3d_gt = selected_gt_3d.reshape(-1, 3)

        '''
        batch_point_corresponds = []
        batch_Ks = []
        batch_Rs = []
        batch_ts = []
        for cam_idx in range(len(self.cam_idxs) - 1):
            # NOTE: C = # cameras = batch dimension --- use batch dim to stack camera pairs.
            # batch_point_corresponds: CxPx2x2
            batch_point_corresponds.append(
                np.stack([point_corresponds[:, 0], point_corresponds[:, cam_idx + 1]], axis=1))
            # batch_Ks: Cx2x3x3
            batch_Ks.append(np.stack([Ks[0], Ks[cam_idx + 1]], axis=0))
            # batch_Ks: Cx2x3x3
            batch_Rs.append(np.stack([Rs[0], Rs[cam_idx + 1]], axis=0))
            # batch_Ks: Cx2x3x1
            batch_ts.append(np.stack([ts[0], ts[cam_idx + 1]], axis=0))
        '''

        point_corresponds = torch.from_numpy(np.array(point_corresponds))
        selected_preds = torch.from_numpy(selected_preds) 
        selected_gt_3d = torch.from_numpy(selected_gt_3d)
        Ks = torch.from_numpy(np.array(Ks))
        Rs = torch.from_numpy(np.array(Rs))
        ts = torch.from_numpy(np.array(ts))

        return point_corresponds, selected_preds, selected_gt_3d, Ks, Rs, ts
