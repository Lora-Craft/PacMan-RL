import torch as t
import torch.nn as nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import Categorical

from model import PMAlpha, PMPolicy, PMValue
from env import make_env

#remove imports below here when done testing
from torchrl.envs import (
    GymWrapper, 
    TransformedEnv,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
)

from torchrl.envs.utils import check_env_specs

is_fork = t.multiprocessing.get_start_method() == "fork"

device = (
    t.device(0)
    if t.cuda.is_available and not is_fork
    else t.device("cpu")
)
num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
total_frames = 1_000_000

sub_batch_size = 64
n_epochs = 5 
clip_epsilon = (0.2)
gamma = 0.99
entropy_eps = 1e-4 

#Shared model backbone for both policy and value head
backbone = PMAlpha()

#Actor head
actor_net = PMPolicy(backbone, num_actions=5)
actor_module = TensorDictModule(
    actor_net,
    in_keys=["pixels"],
    out_keys=["logits"],
)

policy_module = ProbabilisticActor(
    module=actor_module,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=Categorical,
    return_log_prob=True,
)

#Value head
value_net = PMValue(backbone)
value_module = ValueOperator(
    module=value_net,
    in_keys=["pixels"],
)

#GAE and PPO loss
advantage_module = GAE(
    gamma=0.99,
    lmbda=0.95,
    value_network=value_module,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=0.2,
    entropy_bonus=True,
    entropy_coef=1e-4,
)

optimizer = t.optim.Adam(loss_module.parameters(), lr=3e-4)

class PPO:
    """
    PPO implementation for PacMan model
    """
    pass