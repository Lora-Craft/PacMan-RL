import gymnasium as gym

REWARD_CONFIG = { #add in other reward features here when u have decided
    'score_reward': 0.5
}

class ScoreReward:
    """
    Reward based on game score
    """

    def __init__(self):
        self.scale = REWARD_CONFIG['score_reward']
        self.prev_score = 0
    
    def reset(self, info):
        self.prev_score = info.get('score', 0)
    
    def calculate(self, info):
        curr_score = info.get('score', 0)
        d_score = curr_score - self.prev_score
        self.prev_score = curr_score
        return d_score * self.scale



class RewardWrapper(gym.Wrapper):
    """
    Wraps existing environment with defined reward.
    Punishes loss of life
    Maybe you reward cherries here? idk sleep on it 
    """

    def __init__(self, env):
        super().__init__(env)

        self.components = {
            'score': ScoreReward(),
        }
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for comp in self.components.values():
            comp.reset(info)
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        shaped_reward += self.components['score'].calculate(info)

        return obs, shaped_reward, terminated, truncated, info