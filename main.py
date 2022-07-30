import torch
import numpy as np

from src.options import parse_args
from src.run_pose import run as run_pose
from src.infer_pose import infer as infer_pose
from src.run_camera import run as run_camera


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Parse command line args.
    opt, session_id = parse_args()

    if opt.run_mode == 'infer':
        infer_pose(opt)
    else:
        run_pose(opt=opt, session_id=session_id)
