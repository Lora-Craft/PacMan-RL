import torch as t
import torch.nn as nn

from model import PMAlpha
from env import make_env

is_fork = t.multiprocessing.get_start_method() == "fork"

device = (
    t.device(0)
    if t.cuda.is_available and not is_fork
    else t.device("cpu")
)

class PPO:
    """
    PPO implementation for PacMan model
    """
    pass