import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='A model for differentiable RANSAC of autocalibration and 3D human pose.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--hypotheses', '-hyps', type=int, default=50,
        help='number of point correspondences sampled to estimate camera parameters')

    parser.add_argument('--inlierthreshold', '-it', type=float, default=1.,
        help='threshold for 3D line distances used in the soft inlier count')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=0.5,
        help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

    parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0,
        help='scaling factor within the sigmoid of the soft inlier count')

    parser.add_argument('--learningrate', '-lr', type=float, default=0.001,
        help='learning rate')

    parser.add_argument('--lrstep', '-lrs', type=int, default=2500,
        help='cut learning rate in half each x iterations')

    parser.add_argument('--lrstepoffset', '-lro', type=int, default=30000,
        help='keep initial learning rate for at least x iterations')

    parser.add_argument('--trainiterations', '-ti', type=int, default=50000,
        help='number of training iterations (= parameter updates = dataset length)')

    parser.add_argument('--storeinterval', '-si', type=int, default=1000,
        help='store network weights and a prediction vizualisation every x training iterations')

    parser.add_argument('--session', '-sid', default='',
        help='custom session name appended to output files. Useful to separate different runs of the program')

    opt = parser.parse_args()

    if len(opt.session) > 0:
        opt.session = '_' + opt.session
    sid = 'rf%d_c%d_h%d_t%.2f%s' % (
        opt.receptivefield, opt.capacity, opt.hypotheses, opt.inlierthreshold, opt.session)

    return opt
