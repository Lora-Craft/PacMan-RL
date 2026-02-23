import torch as t
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tqdm import tqdm 

from env import make_env
from ppo import make_ppo_mods, HPARAMS
from eval import evaluate

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

    total_batches = HPARAMS["total_frames"] // HPARAMS["frames_per_batch"]
    pbar = tqdm(total=total_batches, desc="Training", unit="batch")

    for batch_idx, tdict_data in enumerate(collector):
        #GAE
        with t.no_grad():
            advantage_module(tdict_data)
        
        adv = tdict_data["advantage"]
        tdict_data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        #PPO minibatch updates
        replay_buffer.extend(tdict_data.reshape(-1)) #Flattens batch

        for epoch in range(HPARAMS["n_epochs"]):
            for minibatch in replay_buffer.sample(
                batch_size=HPARAMS["sub_batch_size"]
            ):
                loss_vals = loss_module(minibatch.to(device))

                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(
                    loss_module.parameters(),
                    HPARAMS["max_grad_norm"],
                )
                optimizer.step()
    
        done_mask = tdict_data["next", "done"]
        ep_reward = tdict_data["next", "episode_reward"][done_mask]

        frames_done = (batch_idx + 1) * HPARAMS["frames_per_batch"]

        if len(ep_reward) > 0:
            mean_reward = ep_reward.mean().item()
            print(f"Batch: {batch_idx}")
            print(f"Frames progress: {frames_done} / {HPARAMS['total_frames']}")
            print(f"Reward: {mean_reward}")
        
        if batch_idx % 50 == 0:
            metrics = evaluate(
                policy_module,
                num_eps=3,
                record_dir=f"recordings/eval_batch_{batch_idx}",
            )
            print(f"Eval reward: {metrics['mean_reward']}")
        
        pbar.update(1)
    
    pbar.close()
    collector.shutdown()
    print("End of training")
