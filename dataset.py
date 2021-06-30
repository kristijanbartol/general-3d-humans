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


class SparseDataset(Dataset):
    '''Temporary dataset class for Human3.6M dataset.'''

    def __init__(self, rootdir, cam_idxs, num_frames=30, num_iterations=50000):
        '''The constructor loads and prepares predictions, GTs, and class parameters.

        rootdir --- the directory where the predictions, camera params and GT are located
        cam_idxs --- the subset of camera indexes used (for now, using 2 cameras)
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

        self.num_iterations = num_iterations
        self.num_frames = num_frames

        # Collect precalculated correspondences, camera params and and 3D GT.
        for dirname in os.listdir(rootdir):
            sidx = int(dirname[1:])
            Ks, Rs, ts = self.__load_camera_params(sidx, cam_idxs)
            self.Ks[sidx] = Ks
            self.Rs[sidx] = Rs
            self.ts[sidx] = ts

            pred_path = os.path.join(rootdir, 'all_2d_preds.npy')
            gt_path = os.path.join(rootdir, 'all_3d_gt.npy')
            bbox_path = os.path.join(rootdir, 'all_bboxes.npy')

            self.preds_2d[sidx] = np.load(pred_path, dtype=np.float32)[:, cam_idxs]
            self.gt_3d[sidx] = np.load(gt_path, dtype=np.float32)
            self.bboxes[sidx] = np.load(bbox_path, dtype=np.float32)[:, cam_idxs]

            # Unbbox keypoints.
            bbox_height = np.abs(self.bboxes[sidx][:, :, 0, 0] - self.bboxes[sidx][:, :, 1, 0])
            self.preds_2d[sidx] *= np.expand_dims(
                np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
            self.preds_2d[sidx] += np.expand_dims(self.bboxes[sidx][:, :, 0, :], axis=2)

            # TODO: Obtain R_rel_gt, t_rel_gt, and scale.

    @staticmethod
    def __load_camera_params(subject_idx, cam_idxs):
        '''Loading camera parameters that are prepared prior to learning.

        Loads camera parameters for a given subject and subset cameras.
        subject_idx --- subject index
        cam_idxs --- subset of camera indexes (currently using 2 cameras)
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

    def __getitem__(self):
        '''
        Get random subset of point correspondences from the preset number of frames.
        '''
        rand_sidx = random.randint(0, len(TRAIN_SIDXS))

        # Selecting a subset of frames.
        selected_frames = np.random.choice(
            np.arange(self.preds_2d[rand_sidx].shape[0]), size=self.num_frames)

        selected_preds = self.preds_2d[rand_sidx][selected_frames]

        # All points stacked along a single dimension for a single subject.
        point_corresponds = np.concatenate(
            np.split(selected_preds, selected_preds.shape[0], axis=0), axis=2)[0].swapaxes(0, 1)

        return point_corresponds, self.gt_3d[rand_sidx], self.Ks[rand_sidx], \
            self.Rs[rand_sidx], self.ts[rand_sidx]
