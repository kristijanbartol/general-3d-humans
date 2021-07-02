import numpy as np
import torch
import kornia
import copy
import cv2
from scipy.spatial.transform import Rotation as R

from mvn.utils.trifocal import Trifocal

#IDXS = [3, 0]
#IDXS = [3, 1, 2]


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = np.array(R).copy()
        assert self.R.shape == (3, 3)

        self.t = np.array(t).copy()
        assert self.t.size == 3
        self.t = self.t.reshape(3, 1)

        self.K = np.array(K).copy()
        assert self.K.shape == (3, 3)

        self.dist = dist
        if self.dist is not None:
            self.dist = np.array(self.dist).copy().flatten()

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    def pixel_in_3d(self, x):
        return torch.inverse(torch.tensor(self.R))

    @property
    def center(self):
        return self.t

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def euclidean_to_homogeneous(points):
    """Converts euclidean points to homogeneous
    Args:
        points numpy array or torch tensor of shape (N, M): N euclidean points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M + 1): homogeneous points
    """
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    elif torch.is_tensor(points):
        return torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)], dim=1)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean
    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M
    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def project_3d_points_to_image_plane_without_distortion(proj_matrix, points_3d, convert_back_to_euclidean=True):
    """Project 3D points to image plane not taking into account distortion
    Args:
        proj_matrix numpy array or torch tensor of shape (3, 4): projection matrix
        points_3d numpy array or torch tensor of shape (N, 3): 3D points
        convert_back_to_euclidean bool: if True, then resulting points will be converted to euclidean coordinates
                                        NOTE: division by zero can be here if z = 0
    Returns:
        numpy array or torch tensor of shape (N, 2): 3D points projected to image plane
    """
    if isinstance(proj_matrix, np.ndarray) and isinstance(points_3d, np.ndarray):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.T
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    elif torch.is_tensor(proj_matrix) and torch.is_tensor(points_3d):
        result = euclidean_to_homogeneous(points_3d) @ proj_matrix.t()
        if convert_back_to_euclidean:
            result = homogeneous_to_euclidean(result)
        return result
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_from_multiple_views_linear_np(proj_matricies, points):
    """Triangulates one point from multiple (N) views using direct linear transformation (DLT).
    For more information look at "Multiple view geometry in computer vision",
    Richard Hartley and Andrew Zisserman, 12.2 (p. 312).
    Args:
        proj_matricies numpy array of shape (N, 3, 4): sequence of projection matricies (3x4)
        points numpy array of shape (N, 2): sequence of points' coordinates
    Returns:
        point_3d numpy array of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)
    A = np.zeros((2 * n_views, 4))
    for j in range(len(proj_matricies)):
        A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
        A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]

    u, s, vh =  np.linalg.svd(A, full_matrices=False)
    point_3d_homo = vh[3, :]

    point_3d = homogeneous_to_euclidean(point_3d_homo)

    return point_3d


def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.view(-1, 4))

    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

    return point_3d


def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None):
    batch_size, n_views, n_joints = points_batch.shape[:3]
    point_3d_batch = torch.zeros(batch_size, n_joints, 3, dtype=torch.float32, device=points_batch.device)

    for batch_i in range(batch_size):
        for joint_i in range(n_joints):
            points = points_batch[batch_i, :, joint_i, :]

            confidences = confidences_batch[batch_i, :, joint_i] if confidences_batch is not None else None
            point_3d = triangulate_point_from_multiple_views_linear_torch(proj_matricies_batch[batch_i], points, confidences=confidences)
            point_3d_batch[batch_i, joint_i] = point_3d

    return point_3d_batch


def triangulate_points_multiview_np(Ks, Rs, ts, points_batch):
    # TODO: Update this in the future.
    BATCH_SIZE = 1
    n_joints, n_views = points_batch.shape[:2]
    points_3d_batch = np.zeros((BATCH_SIZE, n_joints, 3), dtype=np.float32)

    Ps = []
    for view_i in range(n_views):
        extr = np.concatenate((Rs[view_i], ts[view_i]), axis=1)
        Ps.append(Ks[view_i] @ extr)
    Ps = np.stack(Ps, axis=0)

    for batch_i in range(BATCH_SIZE):
        for joint_i in range(n_joints):
            points = points_batch[joint_i]

            point_3d = triangulate_point_from_multiple_views_linear_np(Ps, points)
            points_3d_batch[batch_i, joint_i] = point_3d

    return points_3d_batch


def triangulate_points_multiview_torch(Ks, Rs, ts, points_batch, confidences_batch=None):
    # TODO: Update this in the future.
    BATCH_SIZE = 1
    n_joints, n_views = points_batch.shape[:2]
    points_3d_batch = torch.zeros((BATCH_SIZE, n_joints, 3), dtype=torch.float32, device=points_batch.device)

    Ps = []
    for view_i in range(n_views):
        extr = torch.cat((Rs[view_i], ts[view_i]), axis=1)
        Ps.append(Ks[view_i] @ extr)
    Ps = torch.stack(Ps, dim=0)

    for batch_i in range(BATCH_SIZE):
        for joint_i in range(n_joints):
            points = points_batch[joint_i]
            confidences = confidences_batch[joint_i] if confidences_batch is not None else None

            point_3d = triangulate_point_from_multiple_views_linear_torch(Ps, points, confidences)
            points_3d_batch[batch_i, joint_i] = point_3d

    return points_3d_batch


def calc_reprojection_error_matrix(keypoints_3d, keypoints_2d_list, proj_matricies):
    reprojection_error_matrix = []
    for keypoints_2d, proj_matrix in zip(keypoints_2d_list, proj_matricies):
        keypoints_2d_projected = project_3d_points_to_image_plane_without_distortion(proj_matrix, keypoints_3d)
        reprojection_error = 1 / 2 * np.sqrt(np.sum((keypoints_2d - keypoints_2d_projected) ** 2, axis=1))
        reprojection_error_matrix.append(reprojection_error)

    return np.vstack(reprojection_error_matrix).T


def find_rotation_matrices(points, Ks, device='cuda'):
    K1 = Ks[:, 0]
    K2 = Ks[:, 1]

    conf = torch.ones([1, points.shape[0]], device=device, dtype=torch.float32)
    points = torch.unsqueeze(points, dim=0)
    points1 = points[:, :, 0]
    points2 = points[:, :, 1]
    
    F = kornia.find_fundamental(points1, points2, conf)
    E = kornia.essential_from_fundamental(F, K1, K2)
    R1, R2, t = kornia.decompose_essential_matrix(E)

    return R1, R2, t, F


def find_rotation_matrices_numpy(points, Ks):
    K1 = Ks[0]
    K2 = Ks[1]

    points1 = points[:, 0]
    points2 = points[:, 1]
    
    F, _ = cv2.findFundamentalMat(points1, points2, method=cv2.FM_8POINT, confidence=1.)
    E = K2.T @ F @ K1
    R1, R2, t = cv2.decomposeEssentialMat(E)

    return R1, R2, t, F


def compare_rotations(R_matrices, est_proj_matrices):
    rot_mat1 = R_matrices[0, 0]
    rot_mat2 = R_matrices[0, 1]

    rel_rot = rot_mat2 @ torch.inverse(rot_mat1)

    rel_rot_quat = kornia.rotation_matrix_to_quaternion(rel_rot)
    proj_mat1_quat = kornia.rotation_matrix_to_quaternion(est_proj_matrices[0])
    proj_mat2_quat = kornia.rotation_matrix_to_quaternion(est_proj_matrices[1])

    diff1 = torch.norm(proj_mat1_quat - rel_rot_quat)
    diff2 = torch.norm(proj_mat2_quat - rel_rot_quat)

    return min(diff1, diff2), np.argmin([diff1, diff2])


def compare_rotations_numpy(R_matrices, est_proj_matrices):
    rot_mat1 = R_matrices[0]
    rot_mat2 = R_matrices[1]

    rel_rot = rot_mat2 @ np.linalg.inv(rot_mat1)

    rel_rot_quat = R.from_dcm(rel_rot).as_quat()
    proj_mat1_quat = R.from_dcm(est_proj_matrices[0]).as_quat()
    proj_mat2_quat = R.from_dcm(est_proj_matrices[1]).as_quat()

    diff1 = np.linalg.norm(proj_mat1_quat - rel_rot_quat)
    diff2 = np.linalg.norm(proj_mat2_quat - rel_rot_quat)

    return min(diff1, diff2), np.argmin([diff1, diff2])


def solve_four_solutions(point_corresponds, Ks, Rs, ts, R_cands, t_rel=None):
    K1 = Ks[0]
    K2 = Ks[1]

    R1 = Rs[0]

    t1 = ts[0]
    t2 = ts[1] if t_rel is None else Rs[1] @ torch.inverse(Rs[0]) @ ts[0] + t_rel

    extr1 = torch.cat((R1, t1), dim=1)

#    if t_rel is not None:
    candidate_tuples = [(R_cands[0], t2), (R_cands[0], -t2), (R_cands[1], t2), (R_cands[1], -t2)]
#    else:
#        candidate_tuples = [(R_cands[0], ts[1]), (R_cands[0], -ts[1]), (R_cands[1], ts[1]), (R_cands[1], -ts[1])]

    sign_outcomes = []
    sign_condition = lambda x: torch.all(x[:, 2] > 0.)

    # TODO: Speed up.
    for Rt in candidate_tuples:
        R_rel_est = Rt[0]
        R2_est = R_rel_est @ R1

        extr2_est = torch.cat((R2_est, Rt[1]), dim=1)

        P1 = K1 @ extr1
        P2_est = K2 @ extr2_est

        kpts_3d_est = kornia.geometry.triangulate_points(
            P1, P2_est, point_corresponds[:, 0], point_corresponds[:, 1])
        sign_outcomes.append(sign_condition(kpts_3d_est).cpu())

    return candidate_tuples[sign_outcomes.index(True)]


def solve_four_solutions_numpy(point_corresponds, Ks, Rs, ts, R_cands, t_rel=None):
    K1 = Ks[0]
    K2 = Ks[1]

    R1 = Rs[0]

    t1 = ts[0]
    t2 = ts[1] if t_rel is None else Rs[1] @ np.linalg.inv(Rs[0]) @ ts[0] + t_rel

    extr1 = np.concatenate((R1, t1), axis=1)

#    if t_rel is not None:
    candidate_tuples = [(R_cands[0], t2), (R_cands[0], -t2), (R_cands[1], t2), (R_cands[1], -t2)]
#    else:
#        candidate_tuples = [(R_cands[0], ts[1]), (R_cands[0], -ts[1]), (R_cands[1], ts[1]), (R_cands[1], -ts[1])]

    sign_outcomes = []
    sign_condition = lambda x: np.all(x[:, 2] > 0.)

    # TODO: Speed up.
    for Rt in candidate_tuples:
        R_rel_est = Rt[0]
        R2_est = R_rel_est @ R1

        extr2_est = np.concatenate((R2_est, Rt[1]), axis=1)

        P1 = K1 @ extr1
        P2_est = K2 @ extr2_est

        kpts_3d_est = cv2.triangulatePoints(
            P1, P2_est, point_corresponds[:, 0].T, point_corresponds[:, 1].T).T
        kpts_3d_est = cv2.convertPointsFromHomogeneous(kpts_3d_est).squeeze(axis=1)
        sign_outcomes.append(sign_condition(kpts_3d_est))

    return candidate_tuples[sign_outcomes.index(True)]
    #return candidate_tuples[(sign_outcomes.index(True) + 1) % 4]


def create_fundamental_matrix(Ks, Rs, ts):
    R1 = Rs[:, 0]
    t1 = ts[:, 0]
    R2 = Rs[:, 1]
    t2 = ts[:, 1]

    E = kornia.geometry.essential_from_Rt(R1, t1, R2, t2)

    K1 = Ks[:, 0]
    K2 = Ks[:, 1]

    F = kornia.geometry.fundamental_from_essential(E, K1, K2)

    return F


def project(kpts_3d_gt, K, R, t):
    extr = np.concatenate((R, t), axis=1)
    kpts_3d_gt = np.concatenate((kpts_3d_gt, 
        np.ones((kpts_3d_gt.shape[0], kpts_3d_gt.shape[1], 1))), axis=2)
    kpts_2d_gt = kpts_3d_gt @ np.swapaxes(K @ extr, 0, 1)
    #kpts_2d_gt 

    return kpts_2d_gt


def find_camera_parameters(x1, x2, x3, K1, K2, K3):
    x1 = np.hstack((x1, np.ones((x1.shape[0], 1)))).swapaxes(0, 1)
    x2 = np.hstack((x2, np.ones((x2.shape[0], 1)))).swapaxes(0, 1)
    x3 = np.hstack((x3, np.ones((x3.shape[0], 1)))).swapaxes(0, 1)

    trifocal = Trifocal(x1, x2, x3)
    H21, H31 = trifocal.getProjectionMat(from_E=True)
    #P = trifocal.getProjectionMat(from_E=False)
    F21, F31 = trifocal.getFundamentalMat()

    # TODO: Try swapping/inverting rotations/intrinsics to try all combinations.
    E21 = K1.T.dot(F21).dot(K2)

    U, _, Vt = np.linalg.svd(E21)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)

    R_sol1 = U.dot(W).dot(Vt)
    R_sol2 = U.dot(W.T).dot(Vt)

    #E21 = kornia.essential_from_fundamental(F21, K1, K2)
    #R1, R2, _ = kornia.decompose_essential_matrix(E21)

    #R21 = H21[0, :3, :3]
    #R31 = H31[0, :3, :3]
    #t21 = H21[0, :3, 3]
    #t31 = H31[0, :3, 3]

    #return np.linalg.inv(R21), np.linalg.inv(R31), -t21, -t31
    #return P[:3, :3], P[:3, 3]
    
    return R_sol1, R_sol2, H21[0, :3, :3]


# TODO: Improve parameter ordering and return parameters.
def evaluate_projection(kpts_3d_gt, Ks, Rs, t1, t2, R_rel_est, device='cuda'):
    K1 = Ks[0]
    K2 = Ks[1]

    R1 = Rs[0]
    R2 = Rs[1]
    R2_est = R_rel_est @ R1

    extr1 = torch.cat((R1, t1), dim=1)
    extr2 = torch.cat((R2, t2), dim=1)
    extr2_est = torch.cat((R2_est, t2), dim=1)

    K1 = torch.unsqueeze(K1, dim=0).expand((kpts_3d_gt.shape[0], 3, 3))
    K2 = torch.unsqueeze(K2, dim=0).expand((kpts_3d_gt.shape[0], 3, 3))
    extr1 = torch.unsqueeze(extr1, dim=0).expand((kpts_3d_gt.shape[0], 3, 4))
    extr2 = torch.unsqueeze(extr2, dim=0).expand((kpts_3d_gt.shape[0], 3, 4))
    extr2_est = torch.unsqueeze(extr2_est, dim=0).expand((kpts_3d_gt.shape[0], 3, 4))

    kpts_3d_gt = torch.cat((kpts_3d_gt, torch.ones((kpts_3d_gt.shape[0], kpts_3d_gt.shape[1], 1), device=device)), dim=2)

    kpts_2d_gt2 = kpts_3d_gt @ torch.transpose(K2 @ extr2, 1, 2)
    kpts_2d_gt2 = (kpts_2d_gt2 / kpts_2d_gt2[:, :, 2].reshape(kpts_2d_gt2.shape[0], kpts_2d_gt2.shape[1], 1))[:, :, :2]
    kpts_2d_est = kpts_3d_gt @ torch.transpose(K2 @ extr2_est, 1, 2)
    kpts_2d_est = (kpts_2d_est / kpts_2d_est[:, :, 2].reshape(kpts_2d_est.shape[0], kpts_2d_est.shape[1], 1))[:, :, :2]

    kpts_2d_gt1 = kpts_3d_gt @ torch.transpose(K1 @ extr1, 1, 2)
    kpts_2d_gt1 = (kpts_2d_gt1 / kpts_2d_gt1[:, :, 2].reshape(kpts_2d_gt1.shape[0], kpts_2d_gt1.shape[1], 1))[:, :, :2]

    error_2d = torch.mean(torch.norm(kpts_2d_gt2 - kpts_2d_est, dim=2))

    return torch.stack((kpts_2d_gt1, kpts_2d_gt2), dim=0), error_2d


def evaluate_projection_numpy(kpts_3d_gt, Ks, Rs, ts, R_rel_est, t_rel_est):
    K1 = Ks[0]
    K2 = Ks[1]

    R1 = Rs[0]
    R2 = Rs[1]
    R2_est = R_rel_est @ R1

    t1 = ts[0]
    t2 = t_rel_est if t_rel_est is not None else ts[1]

    extr1 = np.concatenate((R1, t1), axis=1)
    extr2 = np.concatenate((R2, t2), axis=1)
    extr2_est = np.concatenate((R2_est, t2), axis=1)

    K1 = np.expand_dims(K1, axis=0).repeat(kpts_3d_gt.shape[0], axis=0)
    K2 = np.expand_dims(K2, axis=0).repeat(kpts_3d_gt.shape[0], axis=0)
    extr1 = np.expand_dims(extr1, axis=0).repeat(kpts_3d_gt.shape[0], axis=0)
    extr2 = np.expand_dims(extr2, axis=0).repeat(kpts_3d_gt.shape[0], axis=0)
    extr2_est = np.expand_dims(extr2_est, axis=0).repeat(kpts_3d_gt.shape[0], axis=0)

    kpts_3d_gt = np.concatenate((kpts_3d_gt, np.ones((kpts_3d_gt.shape[0], kpts_3d_gt.shape[1], 1))), axis=2)

    kpts_2d_gt2 = kpts_3d_gt @ np.swapaxes(K2 @ extr2, 1, 2)
    kpts_2d_gt2 = (kpts_2d_gt2 / kpts_2d_gt2[:, :, 2].reshape(kpts_2d_gt2.shape[0], kpts_2d_gt2.shape[1], 1))[:, :, :2]
    kpts_2d_est = kpts_3d_gt @ np.swapaxes(K2 @ extr2_est, 1, 2)
    kpts_2d_est = (kpts_2d_est / kpts_2d_est[:, :, 2].reshape(kpts_2d_est.shape[0], kpts_2d_est.shape[1], 1))[:, :, :2]

    kpts_2d_gt1 = kpts_3d_gt @ np.swapaxes(K1 @ extr1, 1, 2)
    kpts_2d_gt1 = (kpts_2d_gt1 / kpts_2d_gt1[:, :, 2].reshape(kpts_2d_gt1.shape[0], kpts_2d_gt1.shape[1], 1))[:, :, :2]

    error_2d = np.mean(np.linalg.norm(kpts_2d_gt2 - kpts_2d_est, axis=2))

    return np.stack((kpts_2d_gt1, kpts_2d_gt2), axis=0), error_2d


# TODO: Improve parameter ordering and return parameters.
def evaluate_reconstruction(kpts_3d_gt, kpts_2d, Ks, Rs, t1, t2, R_rel_est):
    K1 = Ks[0]
    K2 = Ks[1]

    R1 = Rs[0]
    R2 = Rs[1]
    R2_est = R_rel_est @ R1

    extr1 = torch.cat((R1, t1), dim=1)
    extr2 = torch.cat((R2, t2), dim=1)
    extr2_est = torch.cat((R2_est, t2), dim=1)

    P1 = torch.unsqueeze(K1 @ extr1, dim=0).expand((kpts_3d_gt.shape[0], 3, 4))
    P2 = torch.unsqueeze(K2 @ extr2, dim=0).expand((kpts_3d_gt.shape[0], 3, 4))
    P2_est = torch.unsqueeze(K2 @ extr2_est, dim=0).expand((kpts_3d_gt.shape[0], 3, 4))

    kpts_2d_gt1 = kpts_2d[0]
    kpts_2d_gt2 = kpts_2d[1]

    try:
        kpts_3d_est = kornia.geometry.triangulate_points(P1, P2_est, kpts_2d_gt1, kpts_2d_gt2)
    except Exception as ex:
        return None, None

    return torch.mean(torch.norm(kpts_3d_gt - kpts_3d_est, dim=2)), kpts_3d_est


def evaluate_reconstruction_numpy(kpts_3d_gt, kpts_2d, Ks, Rs, ts, R_rel_est, t_rel_est):
    K1 = Ks[0]
    K2 = Ks[1]

    R1 = Rs[0]
    R2 = Rs[1]
    R2_est = R_rel_est @ R1

    t1 = ts[0]
    t2 = t_rel_est if t_rel_est is not None else ts[1]

    extr1 = np.concatenate((R1, t1), axis=1)
    extr2 = np.concatenate((R2, t2), axis=1)
    extr2_est = np.concatenate((R2_est, t2), axis=1)

    P1 = K1 @ extr1
    P2 = K2 @ extr2
    P2_est = K2 @ extr2_est

    kpts_2d_gt1 = kpts_2d[0].reshape(-1, 2).T
    kpts_2d_gt2 = kpts_2d[1].reshape(-1, 2).T

    #kp1_3D = np.ones((3, kpts_2d_gt1.shape[0]))
    #kp2_3D = np.ones((3, kpts_2d_gt2.shape[0]))
    #kp1_3D[0], kp1_3D[1] = kpts_2d_gt1[:, 0].copy(), kpts_2d_gt1[:, 1].copy()
    #kp2_3D[0], kp2_3D[1] = kpts_2d_gt2[:, 0].copy(), kpts_2d_gt2[:, 1].copy()

    kpts_3d_est = cv2.triangulatePoints(P1, P2_est, kpts_2d_gt1, kpts_2d_gt2).T
    #kpts_3d_est = kpts_3d_est[:, :, :3] / kpts_3d_est[:, :, 3, np.newaxis]
    kpts_3d_est = cv2.convertPointsFromHomogeneous(kpts_3d_est).squeeze(axis=1)

    kpts_3d_gt = kpts_3d_gt.reshape(-1, 3)

    return np.mean(np.linalg.norm(kpts_3d_gt - kpts_3d_est, ord=2, axis=1)), kpts_3d_est


def formula(P1, P2, V1, V2):
    a1 = P1[0]
    b1 = P1[1]
    c1 = P1[2]
    a2 = P2[0]
    b2 = P2[1]
    c2 = P2[2]

    a12 = a1 - a2
    b12 = b1 - b2
    c12 = c1 - c2

    p1 = V1[:, 0]
    q1 = V1[:, 1]
    r1 = V1[:, 2]
    p2 = V2[:, 0]
    q2 = V2[:, 1]
    r2 = V2[:, 2]

    return torch.abs(((q1 * r2 - q2 * r1) * a12 + (r1 * p2 - r2 * p1) * b12 + (p1 * q2 - p2 * q1) * c12) \
        / torch.sqrt((q1 * r2 - q2 * r1) ** 2 + (r1 * p2 - r2 * p1) ** 2 + (p1 * q2 - p2 * q1) ** 2))


def formula_np(P1, P2, V1, V2):
    a1 = P1[0]
    b1 = P1[1]
    c1 = P1[2]
    a2 = P2[0]
    b2 = P2[1]
    c2 = P2[2]

    a12 = a1 - a2
    b12 = b1 - b2
    c12 = c1 - c2

    p1 = V1[:, 0]
    q1 = V1[:, 1]
    r1 = V1[:, 2]
    p2 = V2[:, 0]
    q2 = V2[:, 1]
    r2 = V2[:, 2]

    return np.abs(((q1 * r2 - q2 * r1) * a12 + (r1 * p2 - r2 * p1) * b12 + (p1 * q2 - p2 * q1) * c12) \
        / np.sqrt((q1 * r2 - q2 * r1) ** 2 + (r1 * p2 - r2 * p1) ** 2 + (p1 * q2 - p2 * q1) ** 2))


def distance_between_projections(x1, x2, Ks, R1, R_rel, t1, t2, device='cuda'):
    _x1 = torch.cat((x1, torch.ones((x1.shape[0], 1), device=device)), dim=1)
    _x2 = torch.cat((x2, torch.ones((x2.shape[0], 1), device=device)), dim=1)

    R2 = R_rel @ R1

    p1_ = torch.transpose(torch.inverse(R1) @ torch.inverse(Ks[0]) @ torch.transpose(_x1, 0, 1), 0, 1)
    p2_ = torch.transpose(torch.inverse(R2) @ torch.inverse(Ks[1]) @ torch.transpose(_x2, 0, 1), 0, 1)

    return formula(torch.inverse(R1) @ t1[:, 0], torch.inverse(R2) @ t2[:, 0], p1_, p2_)


def distance_between_projections_numpy(x1, x2, Ks, Rs, R_est, t1, t_rel_est):
    K1 = Ks[0]
    K2 = Ks[1]

    R1 = Rs[0]

    _x1 = np.concatenate((x1, np.ones((x1.shape[0], 1))), axis=1)
    _x2 = np.concatenate((x2, np.ones((x2.shape[0], 1))), axis=1)

    R2 = R_est @ R1

    p1_ = np.swapaxes(np.linalg.inv(R1) @ np.linalg.inv(K1) @ np.swapaxes(_x1, 0, 1), 0, 1)
    p2_ = np.swapaxes(np.linalg.inv(R2) @ np.linalg.inv(K2) @ np.swapaxes(_x2, 0, 1), 0, 1)

    return formula_np(np.linalg.inv(R1) @ t1[:, 0], np.linalg.inv(R2) @ t_rel_est[:, 0], p1_, p2_)


def distance_between_line_and_point(a, b, c, x, y):
    return torch.abs(a * x + b * y + c) / torch.sqrt(a ** 2 + b ** 2)


def move_along_epipolar(x1, x2, F):
    epilines = kornia.geometry.compute_correspond_epilines(torch.unsqueeze(x1, dim=0), F)[0]
    assert(torch.all(distance_between_line_and_point(epilines[:, 0], epilines[:, 1], epilines[:, 2], x2[:, 0], x2[:, 1]) < 0.01))

    new_x2 = copy.deepcopy(x2)
    
    new_x2[:, 0] = new_x2[:, 0] + 30.
    new_x2[:, 1] = -(epilines[:, 2] + epilines[:, 0] * new_x2[:, 0]) / epilines[:, 1]

    return new_x2
