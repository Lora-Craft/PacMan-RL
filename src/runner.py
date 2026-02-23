import torch as t
from torchrl.envs.utils import step_mdp

from env import make_env, get_torch_compatible_actions
from model import PMAlpha, PMPolicy, PMValue
from ppo import N_ACTIONS
from train import train

def _get_device():
    is_fork = t.multiprocessing.get_start_method() == "fork"

    device = (
        t.device(0)
        if t.cuda.is_available() and not is_fork
        else t.device("cpu")
    )
    return device


def run_training():
    device = _get_device()
    print(f"Device: {device}")
    train(device=device)

if __name__ == "__main__":
    run_training()





#if __name__ == "__main__":
#    #small test to for model in environment
#    test = make_env()
#    #model = PMAlpha(num_actions=test.action_space.n)
#    backbone = PMAlpha(num_actions=5)
#    policy = PMPolicy(backbone, num_actions=5)
#    value = PMValue(backbone)
#    backbone.eval()
#    tensordict = test.reset()
#    ep_reward = 0.0
#
#    while True:
#        obs = tensordict["pixels"]
#        
#        obs_batch = obs.unsqueeze(0) # adds batch dimensions for model
#        #print(obs)
#        #print(obs_batch)
#        with t.no_grad():
#            #logits, value = model(obs_batch)
#            logits = policy(obs_batch)
#            val = value(obs_batch)
#
#        a_probs = t.softmax(logits, dim=1)
#        action = t.multinomial(a_probs, 1)
#
#        tensordict = test.step(tensordict.set("action", get_torch_compatible_actions(action, num_actions=5)))
#        reward = tensordict["next", "reward"].item()
#        term = tensordict["next", "terminated"]
#        trunc = tensordict["next", "truncated"]
#        ep_reward += reward
#        #print(f"terminated: {term}, truncated: {trunc}")
#    
#        if tensordict["next", "done"].any():
#            ep_length = tensordict["next", "step_count"].item()
#
#        if term or trunc:
#            print("EPISODE ENDED")
#            #print(f"TENSORDICT: {tensordict}")
#            print(ep_reward)
#            print(ep_length)
#            ep_reward = 0.0
#            tensordict = test.reset()
#        
#        else:
#            tensordict = step_mdp(tensordict)