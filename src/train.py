import torch as t
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tqdm import tqdm 

from env import make_env
from ppo import make_ppo_mods, HPARAMS

def train(device=None):
    """
    Training loop using PPO
    """
    if device is None:
        is_fork = t.multiprocessing.get_start_method() == "fork"

        device = (
            t.device(0)
            if t.cuda.is_available() and not is_fork
            else t.device("cpu")
        )
    print(f"Training on: {device}")

    mods = make_ppo_mods(device=device)
    policy_module = mods["policy_module"]
    advantage_module = mods["advantage_module"]
    loss_module = mods["loss_module"]
    optimizer = mods["optimizer"]

    env = make_env()

    #Handles rollout with loop of observe - act - step - store
    #Runs policy_module on current obs to choose action, step in env with the action, stores obs action reward etc, continues until batch has reached frames_per_batch
    collector = SyncDataCollector( 
        env,
        policy_module,
        frames_per_batch=HPARAMS["frames_per_batch"],
        total_frames=HPARAMS["total_frames"],
        split_trajs=False, #If epsiode ends in part way through a batch, collector continues in same batch on next episode
        device=device,
    )

    #Works as minibatch sampler, every batch iteration minibatch of 64 frames taken from 1000 frame input
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=HPARAMS["frames_per_batch"]), 
        sampler=SamplerWithoutReplacement(),
    )


