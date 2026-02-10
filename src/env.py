import retro
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import Resize, Compose, ToTensorImage
from torchvision.transforms import InterpolationMode
import torch as t
from discretizer import Discretizer
from model import PMAlpha

PACMAN_ACTIONS = [
    [],                 #No action
    ['RIGHT'],          #Move right
    ['LEFT'],           #Move left
    ['UP'],             #Move up
    ['DOWN'],           #Move down
]

def get_torch_compatible_actions(actions, num_actions=5):
    return t.nn.functional.one_hot(actions, num_classes=num_actions).float()


def make_env(
        num_envs = 1,
):
    env = retro.make(
            'PacManNamco-Nes',
            render_mode='human',
        )
    
    env = Discretizer(env, PACMAN_ACTIONS)

    wrapped_env = GymWrapper(env)
    
    transformed_env = TransformedEnv(wrapped_env, Compose([
        ToTensorImage(),
        Resize(84, 84, interpolation=InterpolationMode.NEAREST)
    ]))

    return transformed_env


#def disc_wrap(env): #temp function to apply discretizer to env
#    w_env = Discretizer(env, PACMAN_ACTIONS)
#    return w_env

#test = make_env()

#tensordict = test.reset()
#print(f"obs space: {test.observation_space}")
#print(f"obs shape: {test.observation_space.shape}")
#print(f"tdict obs shape: {tensordict[0].shape}")
#print(f"tdict obs dtype: {tensordict[0].dtype}")
# print("ACTION SPEC:", test.action_spec)
#print(tensordict)

#small test to for model in environment
test = make_env()
model = PMAlpha(num_actions=test.action_space.n)
model.eval()
tensordict = test.reset()

while True:
    #action = test.action_space.sample()
    #tensordict = test.step(tensordict.set("action", action))

    obs = tensordict["pixels"]

    obs_batch = obs.unsqueeze(0) # adds batch dimensions for model

    with t.no_grad():
        logits, value = model(obs_batch)
    
    a_probs = t.softmax(logits, dim=1)
    action = t.multinomial(a_probs, 1)

    tensordict = test.step(tensordict.set("action", get_torch_compatible_actions(action, num_actions=5)))
    reward = tensordict["next", "reward"]
    term = tensordict["next", "terminated"]
    trunc = tensordict["next", "truncated"]


    print(f"Action: {action}, Reward: {reward.item():.2f}, Value: {value.item():.4f}")

    if term or trunc:
        print("EPISODE ENDED")
        #print(obs, reward, term, trunc)
        #print(tensordict)
        test.reset()
    

    

