# Author: Kristijan Bartol
import os


H36M_PRETRAINED_PATH = './models/pretrained/h36m_{calib}.pt'
INFER_DATA_DIR = './data/custom/'
INFER_SAVE_DIR = './output/custom/'
INFER_SAVE_PATH = os.path.join(INFER_SAVE_DIR, 'pose_{idx}.npy')

# TODO: Create constants for 'cmu', 'human36m' strings.

KPTS = {
    'lfoot': 0,
    'lknee': 1,
    'lhip': 2,
    'rhip': 3,
    'rknee': 4,
    'rfoot': 5,
    'pelvis': 6,
    'spine': 7,
    'thorax': 8,
    'neck': 9,
    'lwrist': 10,
    'lelbow': 11,
    'lshoulder': 12,
    'rshoulder': 13,
    'relbow': 14,
    'rwrist': 15,
    'head': 16,
}

CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}

# Segment indexes - used for pose prior metrics.
# 0 - head
# 1 - neck
# 2 - left shoulder
# 3 - left upper arm
# 4 - left lower arm
# 5 - right shoulder
# 6 - right upper arm
# 7 - right lower arm
# 8 - back
# 9 - left hip
# 10 - left upper leg
# 11 - left lower leg
# 12 - right hip
# 13 - right upper leg
# 14 - right lower leg
SEGMENT_IDXS = {
    'human36m': [9, 8, 10, 11, 12, 13, 14, 15, 7, 2, 1, 0, 5, 4, 3],    # TODO: h36m uses upper back as a whole back.
    'cmu': [15, 2, 5, 6, 7, 1, 11, 12, 0, 8, 9, 10, 4, 13, 14]   # TODO: cmu uses left eye - head as head.
}
