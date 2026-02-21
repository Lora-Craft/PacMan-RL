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


    

    

