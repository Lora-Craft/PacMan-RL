import torch as t
import torch.nn as nn

from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torch.distributions import OneHotCategorical

from model import PMAlpha, PMPolicy, PMValue
from env import make_env

HPARAMS = {
    "lr": 3e-4,
    "max_grad_norm": 1.0,
    "frames_per_batch": 2048,
    "total_frames": 2_000_000,
    "sub_batch_size": 64,
    "n_epochs": 3,
    "clip_epsilon": 0.2,
    "gamma": 0.99,
    "lambda": 0.95,
    "entropy_eps": 1e-3,
}

N_ACTIONS = 5 #Number of valid actions in action space as defined in env.py

is_fork = t.multiprocessing.get_start_method() == "fork"

device = (
t.device(0)
if t.cuda.is_available() and not is_fork
else t.device("cpu")
)

def make_ppo_mods(device=device):
    """
    Makes all modules used for PPO from torchrl imports and returns each
    module as a key in a dictionary
    """

    #Shared model backbone for both policy and value head
    backbone = PMAlpha(num_actions=N_ACTIONS)

    #Actor head
    actor_net = PMPolicy(backbone, num_actions=N_ACTIONS).to(device)
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["pixels"],
        out_keys=["logits"],
    )

    policy_module = ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )

    #Value head
    value_net = PMValue(backbone).to(device)
    value_module = ValueOperator(
        module=value_net,
        in_keys=["pixels"],
    )

    #GAE and PPO loss
    advantage_module = GAE(
        gamma=HPARAMS["gamma"],
        lmbda=HPARAMS["lambda"],
        value_network=value_module,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=HPARAMS["clip_epsilon"],
        entropy_bonus=True,
        entropy_coeff=HPARAMS["entropy_eps"],
    )

    optimizer = t.optim.Adam(loss_module.parameters(), lr=HPARAMS["lr"])

    return {
        "backbone": backbone,
        "policy_module": policy_module,
        "value_module": value_module,
        "advantage_module": advantage_module,
        "loss_module": loss_module,
        "optimizer": optimizer,
    }
