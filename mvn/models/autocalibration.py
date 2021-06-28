import torch

from mvn.utils.multiview import create_fundamental_matrix, IDXS, find_rotation_matrices, compare_rotations, \
    evaluate_projection, evaluate_reconstruction, distance_between_projections, solve_four_solutions


class Autocalibration():

    def __init__(self, device='cuda'):
        # TODO: camera indexes are temporary. For now, I calibrate only 2 cameras.
        self.camera_idxs = [3, 1]

        self.num_joints = 17
        self.trials = 1000
        self.sample_size = 100
        #self.dist_criterion = .3
        self.num_candidates = int(self.sample_size / 20)

        self.device = device
        
        self.Ks = ...
        self.Rs = ...
        self.ts = ...

        self.point_corresponds = ...
        self.__clear()

    # TODO: Temporary method until autocalibration works for all parameters.
    def set_camera_params(self, Ks, Rs, ts):
        self.Ks = Ks[self.camera_idxs]
        self.Rs = Rs[self.camera_idxs]
        self.ts = ts[self.camera_idxs]

    # TODO: Move to utils.py
    @staticmethod
    def unbbox(preds_2d, bboxes):
        bbox_height = torch.abs(bboxes[:, :, 0, 0] - bboxes[:, :, 1, 0])
        preds_2d *= torch.unsqueeze(torch.unsqueeze(bbox_height / 384., axis=-1), axis=-1)
        preds_2d += torch.unsqueeze(bboxes[:, :, 0, :], axis=2)
        return preds_2d

    def append(self, preds_2d, bboxes):
        preds_2d = preds_2d[:, self.camera_idxs]
        bboxes = bboxes[:, self.camera_idxs]
        preds_2d = self.unbbox(preds_2d, bboxes)

        self.point_corresponds = torch.cat((self.point_corresponds, preds_2d), axis=0)

    def __clear(self):
        # point_corresponds = (num_samples, num_views, num_joints, num_coords)
        self.point_corresponds = torch.empty((0, 2, self.num_joints, 2), device=self.device, dtype=torch.float32)

    def process(self):
        self.point_corresponds = self.point_corresponds.transpose(0, 1).reshape(len(self.camera_idxs), -1, 2)
        num_corresponds = self.point_corresponds.shape[1]
        mean_line_dists = []
        # TODO: device is hardcoded here.
        Rs_rel = torch.empty((0, 3, 3), device=self.device, dtype=torch.float32)

        for i in range(self.trials):
            selected_idxs = torch.multinomial(torch.arange(num_corresponds, dtype=torch.float32), num_samples=self.sample_size, replacement=True)
            R_rel1, R_rel2, _ = find_rotation_matrices(self.point_corresponds[:, selected_idxs], None, torch.unsqueeze(self.Ks, dim=0))

            try:
                R_rel, _ = solve_four_solutions(self.point_corresponds, Ks, Rs, ts, (R_rel1[0], R_rel2[0]))
            except:
                print('Not all positive')
                Rs_rel = torch.cat((Rs_rel, torch.zeros((1, 3, 3), device=self.device, dtype=torch.float32)), axis=0)
                continue

            Rs_rel = torch.cat((Rs_rel, R_rel), axis=0)

            line_dists = distance_between_projections(
                    self.point_corresponds[:, 0], self.point_corresponds[:, 1], 
                    Ks, Rs[0], R_rel, ts)
            mean_line_dists.append(line_dists.mean())

        self.__clear()
        # TODO: Take multiple candidates into account.
        best_idx = torch.tensor(mean_line_dists, device=self.device, dtype=torch.float32).argmin()

        return self.Rs[0, 0] @ Rs_rel[best_idx]
