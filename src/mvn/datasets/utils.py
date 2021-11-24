import numpy as np
import torch

from mvn.utils.img import image_batch_to_torch
from mvn.utils.multiview import Camera

# TODO: Refactor this (first revert and then update make_collate_fn).


def make_collate_fn(randomize_n_views=True, min_n_views=10, max_n_views=31):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['images']) for item in items)

        indexes = np.arange(total_n_views)
        if randomize_n_views:
            n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
            indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
        else:
            indexes = np.arange(total_n_views)

        batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]

        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        # batch['cuboids'] = [item['cuboids'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]

        batch['bbox'] = np.array([[ [[item['bbox'][i][0], item['bbox'][i][1]], [item['bbox'][i][2], item['bbox'][i][3]]] for item in items] for i in indexes], dtype=np.float32).swapaxes(0, 1)

        batch['subject_idx'] = [item['subject_idx'] for item in items]

        # TODO: This is probably why it all became slower in the other branch.
        '''
        try:
            batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
        except:
            pass
        '''

        return batch

    return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def prepare_batch(batch, device, config, is_train=True):
    # images
    images_batch = []
    for image_batch in batch['images']:
        image_batch = image_batch_to_torch(image_batch)
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

    images_batch = torch.stack(images_batch, dim=0)

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]).float().to(device)

    # 3D keypoints validity
    keypoints_3d_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, 3:]).float().to(device)

    # projection matricies
    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
    proj_matricies_batch = proj_matricies_batch.float().to(device)

    labels = np.load('/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy', allow_pickle=True).item()
    frame = labels['table'][0]
    camera_labels = labels['cameras'][frame['subject_idx']]

    # intrinsic matrices
    K_batch = torch.stack([torch.stack([torch.from_numpy(camera.K) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 3)
    K_batch = K_batch.float().to(device)
    Ks = torch.stack([torch.from_numpy(camera_labels['K'][x]) for x in range(4)], dim=0)  # shape (n_views, 3, 3)
    Ks = Ks.float().to(device)

    # rotation matrices
    #R_batch = torch.stack([torch.stack([torch.from_numpy(camera.R) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 3)
    Rs = torch.stack([torch.from_numpy(camera_labels['R'][x]) for x in range(4)], dim=0)  # shape (n_views, 3, 3)
    Rs = Rs.float().to(device)

    # translation vectors
    #t_batch = torch.stack([torch.stack([torch.from_numpy(camera.t) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 3)
    ts = torch.stack([torch.from_numpy(camera_labels['t'][x]) for x in range(4)], dim=0)  # shape (batch_size, n_views, 3, 3)
    ts = ts.float().to(device)

    # bounding boxes
    #bbox_batch = torch.stack([torch.stack([torch.from_numpy(bbox) for bbox in bbox_batch], dim=0) for bbox_batch in batch['bbox']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, bbox_h, bbox_w)
    bbox_batch = torch.stack([torch.from_numpy(bbox_batch) for bbox_batch in batch['bbox']], dim=0)  # shape (batch_size, n_views, bbox_h, bbox_w)
    bbox_batch = bbox_batch.float().to(device)

    subject_idx = batch['subject_idx'][0]       # NOTE: Works only for batch_size=1

    return images_batch, keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt, proj_matricies_batch, Ks, K_batch, Rs, ts, bbox_batch, subject_idx
