import retro
from torchrl.envs import GymWrapper

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
    
    if num_envs == 1:
        wrapped_env = GymWrapper(retro.make(
            'PacManNamco-Nes',
            render_mode='human',
        ))
        
        return wrapped_env

test = make_env()
reset = test.reset()

while True:
    print(reset)
    reset = test.step(reset)

