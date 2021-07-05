import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='A model for differentiable RANSAC of autocalibration and 3D human pose.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--rootdir', type=str, default='./results/',
        help='root directory where 2D predictions, 3D GT and camera params are stored')

    parser.add_argument('--camera_hypotheses', '-chyps', type=int, default=200,
        help='number of sampled hypotheses in every cameraDSAC iteration')

    parser.add_argument('--pose_hypotheses', '-phyps', type=int, default=200,
        help='number of sampled hypotheses in every poseDSAC iteration')

    parser.add_argument('--sample_size', '-ssize', type=int, default=50,
        help='number of point correspondences sampled to estimate camera parameters')

    parser.add_argument('--num_joints', '-jnt', type=int, default=17,
        help='number of joints in human skeleton detector')

    parser.add_argument('--num_frames', type=int, default=50,
        help='number of frames used for camera autocalibration in each iteration')

    parser.add_argument('--num_iterations', '-ti', type=int, default=50000,
        help='number of training iterations (= parameter updates = dataset length)')

    parser.add_argument('--inlier_threshold', '-it', type=float, default=1.,
        help='threshold for 3D line distances used in the soft inlier count')

    parser.add_argument('--inlier_beta', '-ib', type=float, default=100.0,
        help='scaling factor within the sigmoid of the soft inlier count')

    parser.add_argument('--inlier_alpha', '-ia', type=float, default=0.5,
        help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--lr_step', '-lrs', type=int, default=2500,
        help='cut learning rate in half each x iterations')

    parser.add_argument('--lrstepoffset', '-lro', type=int, default=30000,
        help='keep initial learning rate for at least x iterations')

    parser.add_argument('--storeinterval', '-si', type=int, default=1000,
        help='store network weights and a prediction vizualisation every x training iterations')

    parser.add_argument('--use_gt_cameras', dest='use_gt_cameras', action='store_true')

    parser.add_argument('--cpu', dest='cpu', action='store_true')

    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--session', '-sid', default='',
        help='custom session name appended to output files. Useful to separate different runs of the program')

    opt = parser.parse_args()

    if len(opt.session) > 0:
        opt.session = '_' + opt.session
    sid = f'chyps{opt.camera_hypotheses}_cphyps{opt.pose_hypotheses}_ssize{opt.sample_size}_f{opt.num_frames}_{opt.session}'

    return opt, sid
