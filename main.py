import torch
import numpy as np

from src.options import parse_args
from src.learn_pose import run


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Parse command line args.
    opt, session_id = parse_args()

    run(opt=opt, session_id=session_id)
