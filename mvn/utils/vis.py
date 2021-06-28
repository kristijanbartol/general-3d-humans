import numpy as np
import scipy.ndimage
import skimage.transform
import cv2

import torch
import kornia

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from mpl_toolkits import mplot3d
import PIL


from mvn.utils.img import image_batch_to_numpy, to_numpy, denormalize_image, resize_image
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion, \
    find_rotation_matrices, compare_rotations, create_fundamental_matrix, IDXS, evaluate_projection, evaluate_reconstruction

CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}

COLOR_DICT = {
    'coco': [
        (102, 0, 153), (153, 0, 102), (51, 0, 153), (153, 0, 153),  # head
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 102, 0), (153, 153, 0),  # right arm
        (0, 51, 153), (0, 0, 153),  # left leg
        (0, 153, 102), (0, 153, 153),  # right leg
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0)  # body
    ],

    'human36m': [
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # left leg
        (153, 0, 0), (153, 0, 0),  # body
        (153, 0, 102), (153, 0, 102),  # head
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # right arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0)   # left arm
    ],

    'kth': [
        (0, 153, 102), (0, 153, 153),  # right leg
        (0, 51, 153), (0, 0, 153),  # left leg
        (153, 102, 0), (153, 153, 0),  # right arm
        (51, 153, 0), (0, 153, 0),  # left arm
        (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), (153, 0, 0), # body
        (102, 0, 153) # head
    ]
}

JOINT_NAMES_DICT = {
    'coco': {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
}


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image



def visualize_batch(images_batch, heatmaps_batch, keypoints_2d_batch, proj_matricies_batch,
                    K_batch, R_batch,
                    keypoints_3d_batch_gt, keypoints_3d_batch_pred,
                    kind="cmu",
                    cuboids_batch=None,
                    confidences_batch=None,
                    batch_index=0, size=5,
                    max_n_cols=10,
                    pred_kind=None
                    ):
    if pred_kind is None:
        pred_kind = kind

    n_views, n_joints = heatmaps_batch.shape[1], heatmaps_batch.shape[2]

    n_rows = 3
    n_rows = n_rows + 1 if keypoints_2d_batch is not None else n_rows
    n_rows = n_rows + 1 if cuboids_batch is not None else n_rows
    n_rows = n_rows + 1 if confidences_batch is not None else n_rows

    n_cols = min(n_views, max_n_cols)
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * size, n_rows * size))
    axes = axes.reshape(n_rows, n_cols)

    image_shape = images_batch.shape[3:]
    heatmap_shape = heatmaps_batch.shape[3:]

    row_i = 0

    # images
    axes[row_i, 0].set_ylabel("image", size='large')

    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr -> rgb

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
    row_i += 1

    # 2D keypoints (pred)
    if keypoints_2d_batch is not None:
        axes[row_i, 0].set_ylabel("2d keypoints (pred)", size='large')

        keypoints_2d = to_numpy(keypoints_2d_batch)[batch_index]
        for view_i in range(n_cols):
            axes[row_i][view_i].imshow(images[view_i])
            draw_2d_pose(keypoints_2d[view_i], axes[row_i][view_i], kind=kind)
        row_i += 1

    # 2D keypoints (gt projected)
    axes[row_i, 0].set_ylabel("2d keypoints (gt projected)", size='large')

    keypoints_2d_gt_projs = []
    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
        keypoints_2d_gt_proj = project_3d_points_to_image_plane_without_distortion(proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(), keypoints_3d_batch_gt[batch_index].detach().cpu().numpy())
#        keypoints_2d_gt_projs.append(project_3d_points_to_image_plane_without_distortion(proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(), keypoints_3d_batch_gt[batch_index].detach().cpu().numpy()))
        draw_2d_pose(keypoints_2d_gt_proj, axes[row_i][view_i], kind=kind)
#        draw_2d_pose(keypoints_2d_gt_projs[view_i], axes[row_i][view_i], kind=kind)
    row_i += 1

    '''
    keypoints_2d_gt_projs = np.array(keypoints_2d_gt_projs, dtype=np.float32)
    def get_best_points_idxs(kpts_2d, kpts_2d_gt):
        
        def norm(k_pred, k_gt):
            return np.sum(np.abs(k_pred - k_gt), axis=1)
        errors = (norm(kpts_2d[0], kpts_2d_gt[0]) + norm(kpts_2d[1], kpts_2d_gt[1])) / 2.
        best_points_idxs = np.sort(np.argsort(errors)[:10])
        total_error = np.sum(errors[best_points_idxs])
        return best_points_idxs, total_error
    kpts_2d = np.stack((keypoints_2d[IDXS[0]], keypoints_2d[IDXS[1]]))
    kpts_2d_gt = np.stack((keypoints_2d_gt_projs[IDXS[0]], keypoints_2d_gt_projs[IDXS[1]]))
    best_points_idxs, kpt_2d_error = get_best_points_idxs(kpts_2d, kpts_2d_gt)
    kpts_2d_torch = torch.tensor(kpts_2d[:, best_points_idxs], device='cuda', dtype=torch.float32)
    #kpts_2d_torch = torch.tensor(kpts_2d_gt, device='cuda', dtype=torch.float32)
    with torch.no_grad():
        K_batch = torch.unsqueeze(K_batch[batch_index], dim=0)
        R_batch = torch.unsqueeze(R_batch[batch_index], dim=0)
        est_Rs = find_rotation_matrices(kpts_2d_torch, None, K_batch)
        R_sim, m_idx = compare_rotations(R_batch, est_Rs)
    print(f'({kpt_2d_error:.2f}, {R_sim:.2f})')
    '''

    # 2D keypoints (pred projected)
    axes[row_i, 0].set_ylabel("2d keypoints (pred projected)", size='large')

    for view_i in range(n_cols):
        axes[row_i][view_i].imshow(images[view_i])
        keypoints_2d_pred_proj = project_3d_points_to_image_plane_without_distortion(proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(), keypoints_3d_batch_pred[batch_index].detach().cpu().numpy())
        draw_2d_pose(keypoints_2d_pred_proj, axes[row_i][view_i], kind=pred_kind)
    row_i += 1

    # cuboids
    if cuboids_batch is not None:
        axes[row_i, 0].set_ylabel("cuboid", size='large')

        for view_i in range(n_cols):
            cuboid = cuboids_batch[batch_index]
            axes[row_i][view_i].imshow(cuboid.render(proj_matricies_batch[batch_index, view_i].detach().cpu().numpy(), images[view_i].copy()))
        row_i += 1

    # confidences
    if confidences_batch is not None:
        axes[row_i, 0].set_ylabel("confidences", size='large')

        for view_i in range(n_cols):
            confidences = to_numpy(confidences_batch[batch_index, view_i])
            xs = np.arange(len(confidences))

            axes[row_i, view_i].bar(xs, confidences, color='green')
            axes[row_i, view_i].set_xticks(xs)
            if torch.max(confidences_batch).item() <= 1.0:
                axes[row_i, view_i].set_ylim(0.0, 1.0)

    fig.tight_layout()

    fig_image = fig_to_array(fig)

    plt.close('all')

    return fig_image


def draw_epipolar_lines(points, F, img, points2=None, dists=None):
    # lines ... (a, b, c) ... ax + by + c = 0
    lines = kornia.geometry.compute_correspond_epilines(points, F)[0].cpu().numpy()

    start_points = np.zeros((points.shape[1], 2), dtype=np.float32)
    end_points = np.zeros((points.shape[1], 2), dtype=np.float32)

    start_points[:, 0] = 0.
    # (a=0) ... y = -c/b
    start_points[:, 1] = -lines[:, 2] / lines[:, 1]
    end_points[:, 0] = img.shape[0]
    # y = -(c + ax) / b
    end_points[:, 1] = -(lines[:, 2] + lines[:, 0] * end_points[:, 0]) / lines[:, 1]

    for p_idx in range(start_points.shape[0]):
        cv2.line(img, tuple(start_points[p_idx]), tuple(end_points[p_idx]), color=(0, 255, 0))

    if points2 is not None:
        for p_idx, (x, y) in enumerate(points2):
            cv2.putText(img, f'{dists[p_idx]:.1f}', (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

    return lines, img


def draw_images(imgs, name):
    img = np.concatenate(imgs, axis=1)
    cv2.imwrite(f'./results/{name}.png', img)


def my_visualize_batch(candidate_points, iters_total, images_batch, heatmaps_batch, keypoints_2d_batch, proj_matricies_batch,
                    Ks, K_batch, Rs, ts, bbox_batch,
                    keypoints_3d_batch_gt, keypoints_3d_batch_pred,
                    kind="cmu",
                    cuboids_batch=None,
                    confidences_batch=None,
                    batch_index=0, size=5,
                    max_n_cols=10,
                    pred_kind=None
                    ):
    print(f'{iters_total} {batch_index}')
    kpts_2d = torch.stack((keypoints_2d_batch[batch_index][IDXS[0]], keypoints_2d_batch[batch_index][IDXS[1]]), dim=0)
    kpts_2d_bboxed = torch.unsqueeze(kpts_2d.clone(), dim=0)

    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr -> rgb

    img1 = cv2.cvtColor(images[IDXS[0]], cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(images[IDXS[1]], cv2.COLOR_BGR2RGB)
    img_reproj = cv2.cvtColor(images[IDXS[1]], cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        K_batch = torch.unsqueeze(K_batch[batch_index], dim=0)
        Ks = torch.unsqueeze(Ks, dim=0)
        Rs = torch.unsqueeze(Rs, dim=0)
        ts = torch.unsqueeze(ts, dim=0)

        F_created = create_fundamental_matrix(Ks, Rs, ts)
        F_created_bboxed = create_fundamental_matrix(K_batch, Rs, ts)

        # bbox_batch = (B=1, n_views=2, n_points=2, n_coord=2)
        bbox = bbox_batch[batch_index, IDXS]
        bbox_height1 = abs(bbox[0][0][0] - bbox[0][1][0])
        bbox_height2 = abs(bbox[1][0][0] - bbox[1][1][0])

        # TODO: Update magic number.
        kpts_2d[0] *= bbox_height1 / 384.
        kpts_2d[1] *= bbox_height2 / 384.
        kpts_2d[0, :, 0] += bbox[0][0][0]
        kpts_2d[0, :, 1] += bbox[0][0][1]
        kpts_2d[1, :, 0] += bbox[1][0][0]
        kpts_2d[1, :, 1] += bbox[1][0][1]
        # kpts_2d = (B=1, 2, 17, 2)
        kpts_2d = torch.unsqueeze(kpts_2d, dim=0)

        # NOTE: kpts_2d_torch = (B=1, 2, 17, 2)
        # Project epipolar lines from view 0.
        lines = kornia.geometry.compute_correspond_epilines(kpts_2d[:, 0], F_created)[0]
        lines_bboxed = kornia.geometry.compute_correspond_epilines(kpts_2d_bboxed[:, 0], F_created_bboxed)[0]

        # NOTE: dist = |ax + by + c| / sqrt(a ** 2 + b ** 2)
        # Calculate distances between keypoints of view 1 and the epipolar lines from view 0.
        dists = torch.abs(lines[:, 0] * kpts_2d[0, 1, :, 0] + lines[:, 1] * kpts_2d[0, 1, :, 1] + lines[:, 2]) \
            / torch.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)

        dists_bboxed = torch.abs(lines_bboxed[:, 0] * kpts_2d_bboxed[0, 1, :, 0] + lines_bboxed[:, 1] * kpts_2d_bboxed[0, 1, :, 1] + lines_bboxed[:, 2]) \
            / torch.sqrt(lines_bboxed[:, 0] ** 2 + lines_bboxed[:, 1] ** 2)
        
        condition = dists < 0.1
        
        print(f'Number of inliers: {candidate_points.shape[0] + condition.sum()} ({condition.sum()})')

        if candidate_points.shape[0] >= 10:
            R_est1, R_est2, F = find_rotation_matrices(candidate_points, None, Ks)
            R_sim, m_idx = compare_rotations(Rs, (R_est1, R_est2))
            print(f'Quaternion similarity between the rotations: {R_sim:.5f}')
            kpts_3d_gt = keypoints_3d_batch_gt[batch_index]
            kpts_2d_projs = evaluate_projection(kpts_3d_gt, Ks[0], Rs[0], ts[0], R_est1[0] if m_idx == 0 else R_est2[0])

            evaluate_reconstruction(kpts_3d_gt, kpts_2d_projs, Ks[0], Rs[0], ts[0], R_est1[0] if m_idx == 0 else R_est2[0])

            kpts_2d_projs[0, 1, :, 0] -= bbox[1][0][0]
            kpts_2d_projs[0, 1, :, 1] -= bbox[1][0][1]
            kpts_2d_projs[0, 2, :, 0] -= bbox[1][0][0]
            kpts_2d_projs[0, 2, :, 1] -= bbox[1][0][1]
            kpts_2d_projs[0, 1] *= 384. / bbox_height2
            kpts_2d_projs[0, 2] *= 384. / bbox_height2

            draw_2d_pose_cv2(kpts_2d_projs[0, 1].cpu().numpy(), img_reproj)
            draw_2d_pose_cv2(kpts_2d_projs[0, 2].cpu().numpy(), img_reproj)

        new_candidates = torch.transpose(kpts_2d[0, :, condition], 0, 1)
        candidate_points = torch.cat((candidate_points, new_candidates), dim=0)

        epipol_img = draw_2d_pose_cv2(kpts_2d_bboxed[0, 1].cpu().numpy(), np.copy(img2))
        _, epipol_img = draw_epipolar_lines(torch.unsqueeze(kpts_2d_bboxed[0, 0], dim=0), F_created_bboxed, 
            epipol_img, str(iters_total) + str(batch_index), kpts_2d_bboxed[0, 1], dists_bboxed)

        draw_2d_pose_cv2(kpts_2d_bboxed[0, 0].cpu().numpy(), img1)
        draw_2d_pose_cv2(kpts_2d_bboxed[0, 1].cpu().numpy(), img2)

        
        draw_images([img1, img2, epipol_img, img_reproj], f'{iters_total}{batch_index}')

    #print(dists)

    return candidate_points


def visualize_heatmaps(images_batch, heatmaps_batch,
                       kind="cmu",
                       batch_index=0, size=5,
                       max_n_rows=10, max_n_cols=10):
    n_views, n_joints = heatmaps_batch.shape[1], heatmaps_batch.shape[2]
    heatmap_shape = heatmaps_batch.shape[3:]

    n_cols, n_rows = min(n_joints + 1, max_n_cols), min(n_views, max_n_rows)
    fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols * size, n_rows * size))
    axes = axes.reshape(n_rows, n_cols)

    # images
    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr ->

    # heatmaps
    heatmaps = to_numpy(heatmaps_batch[batch_index])

    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0:
                axes[row, col].set_ylabel(str(row), size='large')
                axes[row, col].imshow(images[row])
            else:
                if row == 0:
                    joint_name = JOINT_NAMES_DICT[kind][col - 1] if kind in JOINT_NAMES_DICT else str(col - 1)
                    axes[row, col].set_title(joint_name)

                axes[row, col].imshow(resize_image(images[row], heatmap_shape))
                axes[row, col].imshow(heatmaps[row, col - 1], alpha=0.5)

    fig.tight_layout()

    fig_image = fig_to_array(fig)

    plt.close('all')

    return fig_image


def visualize_volumes(images_batch, volumes_batch, proj_matricies_batch,
                      kind="cmu",
                      cuboids_batch=None,
                      batch_index=0, size=5,
                      max_n_rows=10, max_n_cols=10):
    n_views, n_joints = volumes_batch.shape[1], volumes_batch.shape[2]

    n_cols, n_rows = min(n_joints + 1, max_n_cols), min(n_views, max_n_rows)
    fig = plt.figure(figsize=(n_cols * size, n_rows * size))

    # images
    images = image_batch_to_numpy(images_batch[batch_index])
    images = denormalize_image(images).astype(np.uint8)
    images = images[..., ::-1]  # bgr ->

    # heatmaps
    volumes = to_numpy(volumes_batch[batch_index])

    for row in range(n_rows):
        for col in range(n_cols):
            if col == 0:
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
                ax.set_ylabel(str(row), size='large')

                cuboid = cuboids_batch[batch_index]
                ax.imshow(cuboid.render(proj_matricies_batch[batch_index, row].detach().cpu().numpy(), images[row].copy()))
            else:
                ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection='3d')

                if row == 0:
                    joint_name = JOINT_NAMES_DICT[kind][col - 1] if kind in JOINT_NAMES_DICT else str(col - 1)
                    ax.set_title(joint_name)

                draw_voxels(volumes[col - 1], ax, norm=True)

    fig.tight_layout()

    fig_image = fig_to_array(fig)

    plt.close('all')

    return fig_image


def draw_2d_pose(keypoints, ax, kind='cmu', keypoints_mask=None, point_size=2, line_width=1, radius=None, color=None):
    """
    Visualizes a 2d skeleton
    Args
        keypoints numpy array of shape (19, 2): pose to draw in CMU format.
        ax: matplotlib axis to draw on
    """
    connectivity = CONNECTIVITY_DICT[kind]

    color = 'blue' if color is None else color

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    # points
    ax.scatter(keypoints[keypoints_mask][:, 0], keypoints[keypoints_mask][:, 1], c='red', s=point_size)

    # connections
    for (index_from, index_to) in connectivity:
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            xs, ys = [np.array([keypoints[index_from, j], keypoints[index_to, j]]) for j in range(2)]
            ax.plot(xs, ys, c=color, lw=line_width)

    if radius is not None:
        root_keypoint_index = 0
        xroot, yroot = keypoints[root_keypoint_index, 0], keypoints[root_keypoint_index, 1]

        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])

    ax.set_aspect('equal')


# NOTE: This function was here before, but it wasn't used.
def draw_2d_pose_cv2(keypoints, img, point_size=2, point_color=(255, 255, 255), line_width=1):
    connectivity = CONNECTIVITY_DICT['human36m']

    keypoints_mask = [True] * len(keypoints)

    # connections
    for i, (index_from, index_to) in enumerate(connectivity):
        if keypoints_mask[index_from] and keypoints_mask[index_to]:
            pt_from = tuple(np.array(keypoints[index_from, :]).astype(int))
            pt_to = tuple(np.array(keypoints[index_to, :]).astype(int))

            color = (0, 0, 255)
            cv2.line(img, pt_from, pt_to, color=color, thickness=line_width)

    # points
    for pt in keypoints[keypoints_mask]:
        cv2.circle(img, tuple(pt.astype(int)), point_size, color=point_color, thickness=-1)

    return img


def draw_3d_pose(keypoints, ax, color, keypoints_mask=None, kind='cmu',
        radius=None, root=None, point_size=2, line_width=2, draw_connections=True):
    connectivity = CONNECTIVITY_DICT[kind]

    if keypoints_mask is None:
        keypoints_mask = [True] * len(keypoints)

    if draw_connections:
        # Make connection matrix
        for i, joint in enumerate(connectivity):
            if keypoints_mask[joint[0]] and  keypoints_mask[joint[1]]:
                xs, ys, zs = [np.array([keypoints[joint[0], j], keypoints[joint[1], j]]) for j in range(3)]

                #if kind in COLOR_DICT:
                #    color = COLOR_DICT[kind][i]
                #else:
                #    color = (0, 0, 255)

                #color = np.array(color) / 255

                ax.plot(xs, ys, zs, lw=line_width, c=np.array(color).reshape(1, -1))

        if kind == 'coco':
            mid_collarbone = (keypoints[5, :] + keypoints[6, :]) / 2
            nose = keypoints[0, :]

            xs, ys, zs = [np.array([nose[j], mid_collarbone[j]]) for j in range(3)]

            if kind in COLOR_DICT:
                color = (153, 0, 51)
            else:
                color = (0, 0, 255)

            color = np.array(color) / 255

            ax.plot(xs, ys, zs, lw=line_width, c=color)


    ax.scatter(keypoints[keypoints_mask][:, 0], keypoints[keypoints_mask][:, 1], keypoints[keypoints_mask][:, 2],
               s=point_size, c=np.array([230, 145, 56])/255, edgecolors='black')  # np.array([230, 145, 56])/255

    if radius is not None:
        if root is None:
            root = np.mean(keypoints, axis=0)
        xroot, yroot, zroot = root
        ax.set_xlim([-radius + xroot, radius + xroot])
        ax.set_ylim([-radius + yroot, radius + yroot])
        ax.set_zlim([-radius + zroot, radius + zroot])

    #ax.set_aspect('equal')


    # Get rid of the panes
    background_color = np.array([252, 252, 252]) / 255

    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)

    # Get rid of the ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


def draw_voxels(voxels, ax, shape=(8, 8, 8), norm=True, alpha=0.1):
    # resize for visualization
    zoom = np.array(shape) / np.array(voxels.shape)
    voxels = skimage.transform.resize(voxels, shape, mode='constant', anti_aliasing=True)
    voxels = voxels.transpose(2, 0, 1)

    if norm and voxels.max() - voxels.min() > 0:
        voxels = (voxels - voxels.min()) / (voxels.max() - voxels.min())

    filled = np.ones(voxels.shape)

    # facecolors
    cmap = plt.get_cmap("Blues")

    facecolors_a = cmap(voxels, alpha=alpha)
    facecolors_a = facecolors_a.reshape(-1, 4)

    facecolors_hex = np.array(list(map(lambda x: matplotlib.colors.to_hex(x, keep_alpha=True), facecolors_a)))
    facecolors_hex = facecolors_hex.reshape(*voxels.shape)

    # explode voxels to perform 3d alpha rendering (https://matplotlib.org/devdocs/gallery/mplot3d/voxels_numpy_logo.html)
    def explode(data):
        size = np.array(data.shape) * 2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e

    filled_2 = explode(filled)
    facecolors_2 = explode(facecolors_hex)

    # shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    # draw voxels
    ax.voxels(x, y, z, filled_2, facecolors=facecolors_2)

    ax.set_xlabel("z"); ax.set_ylabel("x"); ax.set_zlabel("y")
    ax.invert_xaxis(); ax.invert_zaxis()


def draw_cameras_and_pose(pose_3d, K1, R1, t1, K2, R2, t2):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    pose_3d_np = pose_3d.cpu().numpy()

    ax.scatter3D(pose_3d_np[:, 0], pose_3d_np[:, 1], pose_3d_np[:, 2], c=pose_3d_np[:, 2], cmap='Greens')

    return fig
