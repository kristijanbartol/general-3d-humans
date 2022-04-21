# Author: Kristijan Bartol


import argparse
import json
from datetime import datetime

from dataset import TRANSFER_CAM_SETS


def parse_args():
    parser = argparse.ArgumentParser(
        description='A model for differentiable RANSAC of autocalibration and 3D human pose.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--rootdir', type=str, default='./results/',
        help='root directory where 2D predictions, 3D GT and camera params are stored')

    #parser.add_argument('--dataset', type=str, default='human36m',
    #    help='the dataset (either human36m or cmu)')

    parser.add_argument('--layers_camdsac', type=int, nargs='+', default=[700, 500, 300, 300, 100],
        help='number of neurons per layer')

    parser.add_argument('--layers_posedsac', type=int, nargs='+', default=[700, 500, 300, 300, 100],
        help='number of neurons per layer')

    parser.add_argument('--cam_idxs', type=int, nargs='+', default=[0, 1, 2, 3])

    parser.add_argument('--camera_hypotheses', '-chyps', type=int, default=100,
        help='number of sampled hypotheses in every cameraDSAC iteration')

    parser.add_argument('--pose_hypotheses', '-phyps', type=int, default=200,
        help='number of sampled hypotheses in every poseDSAC iteration')

    parser.add_argument('--sample_size', '-ssize', type=int, default=80,
        help='number of point correspondences sampled to estimate camera parameters')

    parser.add_argument('--num_joints', '-jnt', type=int, default=17,
        help='number of joints in human skeleton detector')

    parser.add_argument('--num_frames', type=int, default=50,
        help='number of frames used for camera autocalibration in each iteration')

    parser.add_argument('--train_iterations', type=int, default=50,
        help='number of training iterations per epoch (= dataset length)')

    parser.add_argument('--valid_iterations', type=int, default=50,
        help='number of training iterations per epoch (= dataset length)')

    parser.add_argument('--num_epochs', '-e', type=int, default=100,
        help='number of epochs (= num of validations)')

    parser.add_argument('--inlier_threshold', '-it', type=float, default=1.,
        help='threshold for 3D line distances used in the soft inlier count')

    parser.add_argument('--inlier_beta', '-ib', type=float, default=100.0,
        help='scaling factor within the sigmoid of the soft inlier count')

    parser.add_argument('--exp_beta', type=float, default=1.,
        help='the coeficient next to expectation loss component')

    parser.add_argument('--est_beta', type=float, default=0.05,
        help='the coeficient next to error estimation loss component')

    parser.add_argument('--temp', '-t', type=float, default=1.,
        help='softmax temperature regulating how close the distribution is to categorical')

    parser.add_argument('--entropy_beta_cam', '-ebc', type=float, default=1.,
        help='entropy coeficient (the more, the stronger the regularization)')

    parser.add_argument('--entropy_beta_pose', '-ebp', type=float, default=1.,
        help='entropy coeficient (the more, the stronger the regularization)')

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument('--lr_step', '-lrs', type=int, default=100,
        help='cut learning rate in half each x iterations')

    parser.add_argument('--lr_gamma', '-lrg', type=float, default=0.5,
        help='learning rate decay factor')

    parser.add_argument('--temp_step', '-ts', type=int, default=30,
        help='cut learning rate in half each x iterations')

    parser.add_argument('--temp_gamma', '-tg', type=float, default=0.8,
        help='rate of temperature decrease')

    parser.add_argument('--lrstepoffset', '-lro', type=int, default=30000,
        help='keep initial learning rate for at least x iterations')

    parser.add_argument('--storeinterval', '-si', type=int, default=1000,
        help='store network weights and a prediction vizualisation every x training iterations')

    parser.add_argument('--body_lengths_mode', type=int, default=0,
        help='0 - use keypoints only, 1 - along with body lengths, 2 - body lengths only')

    parser.add_argument('--transfer', type=int, default=-1,
        help='use transfer learning and in which mode (-1 no transfer, otherswise pick base camera set')

    parser.add_argument('--pose_batch_size', type=int, default=16,
        help='number of frames after which the gradients are applied')

    parser.add_argument('--posedsac_only', dest='posedsac_only', action='store_true')

    parser.add_argument('--camdsac_only', dest='camdsac_only', action='store_true')

    parser.add_argument('--gumbel', dest='gumbel', action='store_true')

    parser.add_argument('--hard', dest='hard', action='store_true')

    parser.add_argument('--use_body_lengths', dest='use_body_lengths', action='store_true')

    parser.add_argument('--use_estimated', dest='use_estimated', action='store_true')

    parser.add_argument('--entropy_to_scores', dest='entropy_to_scores', action='store_true',
        help='minimize entropy directly on scores, not softmaxed scores')

    parser.add_argument('--weighted_selection', dest='weighted_selection', action='store_true',
        help='weighted selection (regression) instead of probabilistic selection (generative)')

    parser.add_argument('--test', dest='test', action='store_true',
        help='whether to test on test set')

    parser.add_argument('--cpu', dest='cpu', action='store_true')

    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--session', '-sid', default='',
        help='custom session name appended to output files. Useful to separate different runs of the program')

    opt = parser.parse_args()


    session_id = datetime.now().strftime('%d-%b-%Y_%H:%M:%S')
    hyperparams_string = json.dumps(vars(opt), indent=4)

    # NOTE: Number of joints is fixed to 17 (to be able to transfer CMU -> H36M).
    if opt.transfer >= 0:
        opt.cam_idxs = TRANSFER_CAM_SETS[opt.transfer]
        if opt.transfer == 4:
            opt.dataset = 'human36m'
            opt.cam_idxs = TRANSFER_CAM_SETS[4]
        else:   # 0, 1, 2 or 3
            opt.dataset = 'cmu'
    else:
        opt.dataset = 'human36m'
        if opt.posedsac_only:
            opt.cam_idxs = TRANSFER_CAM_SETS[4]     # [0, 1, 2, 3]

    return opt, session_id, hyperparams_string
