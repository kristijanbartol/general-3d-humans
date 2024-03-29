import numpy as np
import torch
import os
import random

from torch.utils.data import Dataset


#TRAIN_SIDXS = [1, 5, 6, 7]
TRAIN_SIDXS = [1]
VAL_SIDXS = [8]
TEST_SIDXS = [9, 11]

TRAIN = 'train'
VALID = 'valid'
TEST = 'test'

SET_SIDXS_MAP = {
    TRAIN: TRAIN_SIDXS,
    VALID: VAL_SIDXS,
    TEST: TEST_SIDXS
}

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


class Human36MDataset(Dataset):
    '''A Dataset class for Human3.6M dataset.'''

    def __init__(self, rootdir, data_type, cam_idxs, custom_dataset=False, num_joints=17, num_frames=30, num_iterations=50):
        '''The constructor loads and prepares predictions, GTs, and class parameters.

        rootdir --- the directory where the predictions, camera params and GT are located
        cam_idxs --- the subset of camera indexes used (the first is used as a reference)
        num_frames --- number of subset frames, M, used (which means P=M*J)
        num_iterations --- number of iterations per epoch, i.e. length of the dataset
        '''
        self.sidxs = SET_SIDXS_MAP[data_type]
        self.data_type = data_type

        # Initialize data for each subject.
        self.Ks = dict.fromkeys(self.sidxs)
        self.Rs = dict.fromkeys(self.sidxs)
        self.ts = dict.fromkeys(self.sidxs)
        self.preds_2d = dict.fromkeys(self.sidxs)
        self.gt_3d = dict.fromkeys(self.sidxs)
        self.bboxes = dict.fromkeys(self.sidxs)

        self.custom_dataset = custom_dataset

        self.cam_idxs = cam_idxs
        self.num_cameras = len(cam_idxs)
        self.num_iterations = num_iterations
        self.num_joints = num_joints
        self.num_frames = num_frames

        # NOTE: Set CamDSAC flag in case CamDSAC is tested (different loading).
        self.cam_dsac = False
        #self.cam_dsac = False
        #if self.num_iterations is None:
        #    self.num_iterations = 20
        #    self.cam_dsac = True

        self.num_poses = 0

        # Collect precalculated correspondences, camera params and and 3D GT,
        # for every subject, based on folder indexes.
        for dirname in os.listdir(rootdir):
            sidx = int(dirname[1:]) if dirname[0] == 'S' else -1
            if not sidx in self.sidxs:
                continue
            Ks, Rs, ts = self.__load_camera_params(sidx, cam_idxs, self.custom_dataset)
            self.Ks[sidx] = Ks
            self.Rs[sidx] = Rs
            self.ts[sidx] = ts

            dirpath = os.path.join(rootdir, dirname)

            self.preds_2d[sidx] = np.empty((0, self.num_cameras, num_joints, 2), dtype=np.float32)
            self.gt_3d[sidx] = np.empty((0, num_joints, 3), dtype=np.float32)
            self.bboxes[sidx] = np.empty((0, self.num_cameras, 2, 2), dtype=np.float32)
            fcounter = 0

            all_gt_3d = np.empty((0, 3), dtype=np.float32)

            while True:
                pred_path = os.path.join(dirpath, f'all_2d_preds{fcounter}.npy')
                gt_path = os.path.join(dirpath, f'all_3d_gt{fcounter}.npy')
                bbox_path = os.path.join(dirpath, f'all_bboxes{fcounter}.npy')

                if not os.path.exists(pred_path):
                    # All data has been collected.
                    break

                preds_2d = np.load(pred_path)[:, cam_idxs]
                gt_3d = np.load(gt_path)
                bboxes = np.load(bbox_path)[:, cam_idxs]

                self.preds_2d[sidx] = np.concatenate((self.preds_2d[sidx], preds_2d), axis=0)
                self.gt_3d[sidx] = np.concatenate((self.gt_3d[sidx], gt_3d), axis=0)
                self.bboxes[sidx] = np.concatenate((self.bboxes[sidx], bboxes), axis=0)

                all_gt_3d = np.concatenate((all_gt_3d, gt_3d.reshape((-1, 3))), axis=0)

                self.num_poses += self.gt_3d[sidx].shape[0]
                fcounter += 1

            # Unbbox keypoints.
            bbox_height = np.abs(self.bboxes[sidx][:, :, 0, 0] - self.bboxes[sidx][:, :, 1, 0])
            self.preds_2d[sidx] *= np.expand_dims(
                np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
            self.preds_2d[sidx] += np.expand_dims(self.bboxes[sidx][:, :, 0, :], axis=2)

        self.mean_3d = np.mean(all_gt_3d, axis=0)
        self.std_3d = np.std(all_gt_3d, axis=0)

        # TODO: Obtain GT scale to estimate translation also.

    @staticmethod
    def __load_camera_params(subject_idx, cam_idxs, custom_dataset=False):
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

        if custom_dataset:
            est_Rs = np.load('./results/est_Rs.npy')
            est_ts = np.load('./results/est_ts.npy')

            for cam_idx in cam_idxs:
                Rs[cam_idx] = est_Rs[cam_idx] @ Rs[0]
                if cam_idx > 0:
                    ts[cam_idx] = est_ts[cam_idx]


        return Ks, Rs, ts

    def __len__(self):
        #if self.data_type == TEST:
        #    # 40
        #    return (self.preds_2d[9].shape[0] + self.preds_2d[11].shape[0]) // self.num_frames + 1
        #else:
        #    return self.num_iterations
        return self.num_iterations

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
        if self.data_type == TEST and not self.cam_dsac:
            first_frame = idx * self.num_frames
            last_frame = (idx + 1) * self.num_frames
            subject_nine_frames = self.preds_2d[9].shape[0]
            sidx = 9 if last_frame < subject_nine_frames else 11
            first_frame = first_frame if sidx == 9 else first_frame - subject_nine_frames
            last_frame = last_frame if sidx == 9 else min(last_frame - subject_nine_frames, self.preds_2d[11].shape[0])

            selected_frames = np.arange(first_frame, last_frame)
        else:
            rand_idx = random.randint(0, len(self.sidxs) - 1)
            sidx = self.sidxs[rand_idx]

            # Selecting a subset of frames.
            selected_frames = np.random.choice(
                np.arange(self.preds_2d[sidx].shape[0]), size=self.num_frames)

        # Select 2D predictions, 3D GT, and camera parameters 
        # for a given random subject and selected frames.
        selected_preds = self.preds_2d[sidx][selected_frames]
        selected_gt_3d = self.gt_3d[sidx][selected_frames]
        Ks = self.Ks[sidx]
        Rs = self.Rs[sidx]
        ts = self.ts[sidx]

        # All points stacked along a single dimension for a single subject.
        point_corresponds = np.concatenate(
            np.split(selected_preds, selected_preds.shape[0], axis=0), axis=2)[0]

        point_corresponds = torch.from_numpy(np.array(point_corresponds))
        selected_preds = torch.from_numpy(selected_preds) 
        selected_gt_3d = torch.from_numpy(selected_gt_3d)
        Ks = torch.from_numpy(np.array(Ks))
        Rs = torch.from_numpy(np.array(Rs))
        ts = torch.from_numpy(np.array(ts))

        return point_corresponds, selected_preds, selected_gt_3d, Ks, Rs, ts


CMU_TO_H36M_MAP = (
    (0, 1, 2,  3,  4,  5, 6, 8, 9, 10, 11, 12, 13, 14, 15),     # Human36M
    (8, 7, 6, 12, 13, 14, 2, 0, 1, 5,  4,  3,  9, 10, 11)      # CMUPanoptic
)

# Transfer learning mode - 0
CAM_SET_CONFIG = {
    '0': [1, 2, 3, 4, 6, 7, 10],
    '1': [12, 16, 18, 19, 22, 23, 30]
}

# Transfer learning mode - 1
CAM_SET_NUMBER = {
#    '0': [1, 2, 3, 4, 6, 7],
    '1': [10, 12, 16, 18],
    '2': [10, 12, 16, 18, 19, 22, 23, 30]
}

# Transfer learning mode - 2
#CAM_SET_DATASET = {
#    '0': [1, 2, 3, 4, 6, 7, 10, 12]
#}

TRANSFER_CAM_SETS = [
    [1, 2, 3, 4, 6, 7, 10],                     # 7 CMU cameras
    [12, 16, 18, 19, 22, 23, 30],               # other 7 CMU cameras
    [10, 12, 16, 18],                           # 4 CMU cameras
    [6, 7, 10, 12, 16, 18, 19, 22, 23, 30],     # 10 CMU cameras

    [0, 1, 2, 3],                               # H36M cameras

    # Supplementary
    #list(range(31)),
    [1, 2, 3, 4, 6, 7, 10, 12, 16, 18, 19, 22, 23, 30],
    [0, 5, 8, 9, 11, 13, 14],
    [15, 17, 20, 21, 24, 25, 26, 27, 28, 29],
    [0, 5, 8, 9, 11, 13, 14, 15, 17, 20, 21, 24, 25, 26, 27, 28, 29]
]


class CmuPanopticDataset(Dataset):

    def __init__(self, rootdir, data_type, cam_idxs, custom_dataset=False, num_joints=19, num_frames=30, num_iterations=50):
        self.rootdir = rootdir
        self.data_type = data_type
        self.cam_idxs = cam_idxs
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.num_iterations = num_iterations

        self.num_views = len(self.cam_idxs)

        self.custom_dataset = custom_dataset

        # Load data.
        self.Ks, self.Rs, self.ts = self.__load_camera_params()
        # TODO: Improve camera index selection.
        #self.preds_2d = np.load(os.path.join(self.rootdir, 'all_2d_preds.npy'))[:, -10:]
        self.preds_2d = np.load(os.path.join(self.rootdir, 'all_2d_preds.npy'))[:, self.cam_idxs]
        self.gt_3d = np.load(os.path.join(self.rootdir, 'all_3d_gt.npy'))
        #self.bboxes = np.load(os.path.join(self.rootdir, 'all_bboxes.npy'))[:, -10:].reshape(
        self.bboxes = np.load(os.path.join(self.rootdir, 'all_bboxes.npy'))[:, self.cam_idxs].reshape(
            (-1, self.num_views, 2, 2))

        self.num_samples = self.preds_2d.shape[0]

        # CMU to H36M.
        self.gt_3d = self.__cmu_to_h36m(self.gt_3d)

        # Unbbox keypoints.
        bbox_height = np.abs(self.bboxes[:, :, 0, 0] - self.bboxes[:, :, 1, 0])
        self.preds_2d *= np.expand_dims(
            np.expand_dims(bbox_height / 384., axis=-1), axis=-1)
        self.preds_2d += np.expand_dims(self.bboxes[:, :, 0, :], axis=2)

        # Dataset statistics (for normalization).
        self.mean_3d = np.mean(self.gt_3d, axis=0)
        self.std_3d = np.std(self.gt_3d, axis=0)

    def __load_camera_params(self):
        '''Loading camera parameters for given subject and camera subset.

        Loads camera parameters for a given subject and subset cameras.
        subject_idx --- subject index
        cam_idxs --- subset of camera indexes
        '''
        labels = np.load('/data/cmu/cmu-multiview-labels-MRCNNbboxes.npy', 
            allow_pickle=True).item()
        camera_params = labels['cameras'][0]

        Ks, Rs, ts = [], [], []
        for cam_idx in self.cam_idxs:
            Ks.append(camera_params[cam_idx][2])
            Rs.append(camera_params[cam_idx][0])
            ts.append(camera_params[cam_idx][1])

        Ks = np.stack(Ks, axis=0)
        Rs = np.stack(Rs, axis=0)
        ts = np.stack(ts, axis=0)

        return Ks, Rs, ts

    @staticmethod
    def __cmu_to_h36m(cmu_kpts):
        h36m_kpts = np.empty((cmu_kpts.shape[0], 17, cmu_kpts.shape[2]), dtype=np.float32)
        h36m_kpts[:, CMU_TO_H36M_MAP[0], :] = cmu_kpts[:, CMU_TO_H36M_MAP[1], :]
        h36m_kpts[:, 7, :] = np.mean(cmu_kpts[:, [0, 2], :], axis=1)
        h36m_kpts[:, 16, :] = np.mean(cmu_kpts[:, [15, 16, 17, 18], :], axis=1)
        return h36m_kpts

    def __len__(self):
        return self.num_iterations

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
        # Selecting a subset of frames.
        if self.data_type == TRAIN:
            selected_frames = np.random.choice(
                np.arange(int(self.num_samples * 0.6)), 
                size=self.num_frames
            )
        elif self.data_type == VALID:
            selected_frames = np.random.choice(
                np.arange(int(self.num_samples * 0.6) + 1, int(self.num_samples * 0.8)), 
                size=self.num_frames)
        elif self.data_type == TEST:
            selected_frames = np.random.choice(
                np.arange(int(self.num_samples * 0.8) + 1, self.num_samples), 
                size=self.num_frames)

        # Select 2D predictions, 3D GT, and camera parameters 
        # for a given random subject and selected frames.
        selected_preds = self.preds_2d[selected_frames]
        selected_gt_3d = self.gt_3d[selected_frames]

        # All points stacked along a single dimension for a single subject.
        point_corresponds = np.concatenate(
            np.split(selected_preds, selected_preds.shape[0], axis=0), axis=2)[0]

        point_corresponds = torch.from_numpy(np.array(point_corresponds))
        selected_preds = torch.from_numpy(selected_preds) 
        selected_gt_3d = torch.from_numpy(selected_gt_3d)
        Ks = torch.from_numpy(self.Ks)
        Rs = torch.from_numpy(self.Rs)
        ts = torch.from_numpy(self.ts)

        return point_corresponds, selected_preds, selected_gt_3d, Ks, Rs, ts



def init_datasets(opt):
    data_rootdir = f'./data/{opt.dataset}'
    dataset = Human36MDataset if opt.dataset == 'human36m' else CmuPanopticDataset

    cam_test_set = []
    test_sets = []

    if opt.transfer == -1:
        cam_test_set.append(opt.cam_idxs)
    else:
        #for set_idx in range(5):
            # NOTE: Testing also on base set.
        #    cam_test_set.append(TRANSFER_CAM_SETS[set_idx])
        cam_test_set.append(TRANSFER_CAM_SETS[opt.transfer])

    train_set = dataset(data_rootdir, TRAIN, opt.cam_idxs, custom_dataset=False, num_joints=opt.num_joints, 
        num_frames=opt.num_frames, num_iterations=opt.train_iterations)
    valid_set = dataset(data_rootdir, VALID, opt.cam_idxs, custom_dataset=False, num_joints=opt.num_joints, 
        num_frames=opt.num_frames, num_iterations=opt.valid_iterations)

    # TODO: Update magic number.
    test_set = Human36MDataset('./data/human36m', TEST, [0, 1, 2, 3], custom_dataset=opt.custom_dataset, 
        num_joints=opt.num_joints, num_frames=opt.num_frames, num_iterations=40)

    return train_set, valid_set, [test_set]



class CustomTestDataset(Dataset):
    '''A Dataset class for custom test dataset.'''

    def __init__(self, kpts, Rs, ts, mean, std, num_joints=17):
        self.kpts = kpts
        self.num_joints = num_joints

        # Use intrinsics from H36M, for now.
        labels = np.load('/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', 
            allow_pickle=True).item()
        camera_params = labels['cameras'][SUBJECT_ORD_MAP[1]]

        self.K = camera_params[0][2]
        self.Rs = Rs
        self.ts = ts
        self.mean = mean
        self.std = std
        
        self.num_views = self.kpts.shape[1]

    def __len__(self):
        return self.kpts.shape[0]

    def __getitem__(self, idx):
        frame_kpts = self.kpts[idx]

        # All points stacked along a single dimension for a single subject.
        point_corresponds = np.concatenate(
            np.split(frame_kpts, frame_kpts.shape[0], axis=0), axis=2)[0]

        point_corresponds = torch.from_numpy(np.array(point_corresponds))
        frame_kpts = torch.from_numpy(frame_kpts) 
        Ks = torch.from_numpy(np.array(self.K)).unsqueeze(0).tile(self.num_views, 1, 1)
        Rs = torch.from_numpy(np.array(self.Rs))
        ts = torch.from_numpy(np.array(self.ts))

        return frame_kpts, Ks, Rs, ts
