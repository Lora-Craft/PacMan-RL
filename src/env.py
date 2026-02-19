import retro
from torchrl.envs import (
    GymWrapper, 
    TransformedEnv,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    RewardSum,
)
from torchrl.envs.transforms import Resize, Compose, ToTensorImage
from torchvision.transforms import InterpolationMode
from torchrl.envs.utils import step_mdp
import torch as t
from discretizer import Discretizer
from model import PMAlpha, PMPolicy, PMValue
from reward import RewardWrapper

PACMAN_ACTIONS = [
    [],                 #No action
    ['RIGHT'],          #Move right
    ['LEFT'],           #Move left
    ['UP'],             #Move up
    ['DOWN'],           #Move down
]

def get_torch_compatible_actions(actions, num_actions=5):
    """
    Torchrl expects actions to be one-hot encoded
    Function converts integer actions into one-hot matrix
    """
    return t.nn.functional.one_hot(actions, num_classes=num_actions).float()


def make_env(
        num_envs = 1,
):
    env = retro.make(
            'PacManNamco-Nes',
            render_mode='human',
        )
    
    env = Discretizer(env, PACMAN_ACTIONS)

    env = RewardWrapper(env)

    wrapped_env = GymWrapper(env)
    
    transformed_env = TransformedEnv(wrapped_env, Compose([
        ToTensorImage(),
        Resize(84, 84, interpolation=InterpolationMode.NEAREST),
        #ObservationNorm(in_keys=["observation"]),
        #DoubleToFloat(),
        StepCounter(),
        RewardSum(),
    ]))

    return transformed_env

if __name__ == "__main__":
    #small test to for model in environment
    test = make_env()
    #model = PMAlpha(num_actions=test.action_space.n)
    backbone = PMAlpha(num_actions=5)
    policy = PMPolicy(backbone, num_actions=5)
    value = PMValue(backbone)
    backbone.eval()
    tensordict = test.reset()
    ep_reward = 0.0

    while True:
        obs = tensordict["pixels"]
        
        obs_batch = obs.unsqueeze(0) # adds batch dimensions for model
        #print(obs)
        #print(obs_batch)
        with t.no_grad():
            #logits, value = model(obs_batch)
            logits = policy(obs_batch)
            val = value(obs_batch)

        a_probs = t.softmax(logits, dim=1)
        action = t.multinomial(a_probs, 1)

        tensordict = test.step(tensordict.set("action", get_torch_compatible_actions(action, num_actions=5)))
        reward = tensordict["next", "reward"].item()
        term = tensordict["next", "terminated"]
        trunc = tensordict["next", "truncated"]
        ep_reward += reward
        #print(f"terminated: {term}, truncated: {trunc}")
    
        if tensordict["next", "done"].any():
            ep_length = tensordict["next", "step_count"].item()

        if term or trunc:
            print("EPISODE ENDED")
            #print(f"TENSORDICT: {tensordict}")
            print(ep_reward)
            print(ep_length)
            ep_reward = 0.0
            tensordict = test.reset()
        
        else:
            tensordict = step_mdp(tensordict)
    

    

