import torch as t
import os
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from tqdm import tqdm 

from env import make_env
from ppo import make_ppo_mods, HPARAMS
from eval import evaluate

def save_checkpoint(mods, batch_idx, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "backbone_state": mods["backbone"].state_dict(),
        "policy_state": mods["policy_module"].state_dict(),
        "value_state": mods["value_module"].state_dict(),
        "optimizer_state": mods["optimizer"].state_dict(),
        "batch_idx": batch_idx,
        "hparams": HPARAMS,
    }

    model_path = os.path.join(checkpoint_dir, f"checkpoint_batch_{batch_idx}.pt")
    t.save(checkpoint, model_path)
    print(f"Checkpoint saved: {model_path}")

    return model_path

def load_checkpoint(checkpoint_path, mods, device="cpu"):
    checkpoint = t.load(checkpoint_path, map_location=device)

    mods["backbone"].load_state_dict(checkpoint["backbone_state"])
    mods["policy_module"].load_state_dict(checkpoint["policy_state"])
    mods["value_module"].load_state_dict(checkpoint["value_state"])
    mods["optimizer"].load_state_dict(checkpoint["optimizer_state"])

    start_batch = checkpoint["batch_idx"] + 1
    print(f"Loaded from batch: {checkpoint['batch_idx']}")

    return start_batch

def train(device=None, checkpoint_path=None):
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

    start_batch = 0
    if checkpoint_path is not None:
        start_batch = load_checkpoint(checkpoint_path, mods, device=device)

    env = make_env()

    #Adjusts training frames for checkpoint
    remaining_frames = HPARAMS["total_frames"] - (start_batch * HPARAMS["frames_per_batch"])

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

    checkpoint_freq = 50 #Checkpoint after 50 batches

    for batch_idx, tdict_data in enumerate(collector):
        actual_batch = start_batch + batch_idx

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

        frames_done = (actual_batch + 1) * HPARAMS["frames_per_batch"]

        if len(ep_reward) > 0:
            mean_reward = ep_reward.mean().item()
            print(f"Batch: {actual_batch}")
            print(f"Frames progress: {frames_done} / {HPARAMS['total_frames']}")
            print(f"Reward: {mean_reward}")

        #Checkpoint conditional
        if actual_batch % checkpoint_freq == 0 and actual_batch > 0:
            save_checkpoint(mods, actual_batch)
        
        #if batch_idx % 50 == 0:
        #    metrics = evaluate(
        #        policy_module,
        #        num_eps=3,
        #        record_dir=f"recordings/eval_batch_{batch_idx}",
        #    )
        #    print(f"Eval reward: {metrics['mean_reward']}")
        
        pbar.update(1)
    
    #Save final checkpoint
    save_checkpoint(mods, actual_batch, checkpoint_dir="checkpoints")

    pbar.close()
    collector.shutdown()
    print("End of training")
