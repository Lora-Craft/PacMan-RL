import retro
from torchrl.envs import (
    GymWrapper, 
    TransformedEnv,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    RewardSum,
)
from gymnasium.wrappers import RecordVideo
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


def make_env(render=False, record=False, record_dir=None):
    env = retro.make(
            'PacManNamco-Nes',
            render_mode='human' #if render else 'rgb_array',
        )
    
    env = Discretizer(env, PACMAN_ACTIONS)
    env = RewardWrapper(env)

    if record and record_dir is not None:
        env = RecordVideo(
            env,
            video_folder=record_dir,
            episode_trigger=lambda x: True,
            name_prefix="pacman_eval",
        )

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


    

    

