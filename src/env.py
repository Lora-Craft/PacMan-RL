import retro
from torchrl.envs import GymWrapper
import torch as t
from discretizer import Discretizer

PACMAN_ACTIONS = [
    [],                 #No action
    ['RIGHT'],          #Move right
    ['LEFT'],           #Move left
    ['UP'],             #Move up
    ['DOWN'],           #Move down
]

def make_env(
        num_envs = 1,
):
    return retro.make(
            'PacManNamco-Nes',
            render_mode='human',
        )
        

    if num_envs == 1:
        wrapped_env = GymWrapper(retro.make(
            'PacManNamco-Nes',
            render_mode='human',
        ))
        
        return wrapped_env

def disc_wrap(env): #temp function to apply discretizer to env
    w_env = Discretizer(env, PACMAN_ACTIONS)
    return w_env

test = make_env()
test = disc_wrap(test)
#test.action_space = t.zeros(9)
tensordict = test.reset()
# print("ACTION SPEC:", test.action_spec)

while True:
    action = test.action_space.sample()

    #action = test.action_space
    #tensordict = test.step(action)
    obs, reward, term, truncated, info = test.step(action)
    print("Lives:", info.get('lives', 'NOT FOUND'))
    print("Done:", term)
    print("Info:", info)
    
    if term:
        print("EPISODE ENDED")
        test.reset()
    

    

