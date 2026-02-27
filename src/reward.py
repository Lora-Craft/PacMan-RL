import gymnasium as gym

REWARD_CONFIG = { #add in other reward features here when u have decided
    'score_reward': 1.0, #In game score has a 0 hard-coded onto the end, actual score is thus 1/10th of displayed
    'death_penalty': 2.0,
    'step_penalty': 0.002, #Penalises each step to force agent to find reward
}             


class ScoreReward:
    """
    Reward based on game score
    """

    def __init__(self):
        self.scale = REWARD_CONFIG['score_reward']
        self.prev_score = 0
    
    def reset(self, info):
        """
        Info dictionary contains variabls from the game
        and their RAM address. Provided by retro. 
        """
        self.prev_score = info.get('score', 0)
    
    def calculate(self, info):
        curr_score = info.get('score', 0)
        d_score = curr_score - self.prev_score
        self.prev_score = curr_score
        return d_score * self.scale

class DeathReward:
    """
    Punishes loss of life
    """

    def __init__(self):
        self.penalty = REWARD_CONFIG['death_penalty']
        self.last_state = 3
    
    def reset(self, info): # check if you need to keep this to match API signature or if it can be removed 
        self.last_state = info.get('lives', 3)
    
    def calculate(self, info):
        curr_state = info.get('lives', 0)
        reward = 0.0
        
        if self.last_state is not None and curr_state < self.last_state:
            reward += self.penalty
        #if curr_state < self.last_state:
        #    reward -= self.penalty
        #    self.last_state = curr_state
        #
        #if curr_state == 0:
        #    self.last_state = 3
        #print(f"CURRENT STATE IS: {curr_state}")
        #print(f"PREV STATE IS: {self.last_state}")
        self.last_state = curr_state
        return reward

class RewardWrapper(gym.Wrapper):
    """
    Wraps existing environment with defined reward.
    Punishes loss of life (TBD)
    Maybe you reward cherries here? idk sleep on it
    """

    def __init__(self, env):
        super().__init__(env)

        self.step_penalty = REWARD_CONFIG['step_penalty']
        self.components = {
            'score': ScoreReward(),
            'lives': DeathReward(),
        }
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for comp in self.components.values():
            comp.reset(info)
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0 #Look at this later

        shaped_reward += self.components['score'].calculate(info)
        shaped_reward -= self.components['lives'].calculate(info)
        shaped_reward -= self.step_penalty

        return obs, shaped_reward, terminated, truncated, info