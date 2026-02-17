import gymnasium as gym

class RewardWrapper(gym.Wrapper):
    """
    Wraps existing environment with defined reward.
    Punishes loss of life
    Maybe you reward cherries here? idk sleep on it 
    """