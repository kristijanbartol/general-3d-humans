import torch
import numpy as np

from src.options import parse_args
from src.learn_pose import run_pose


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    # Parse command line args.
    opt, session_id, hyperparams_string = parse_args()

    run_pose(
        opt=opt, 
        session_id=session_id, 
        hyperparams_string=hyperparams_string)
