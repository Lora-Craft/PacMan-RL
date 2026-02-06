import numpy as np
import gymnasium as gym

class Discretizer(gym.ActionWrapper):
    """
    Wraps the gym environment to use combinations of defined moves for model actions
    Sets all buttons other than the button that should currently be pressed as False
    and sets the button that should be currently pressed as True.
    """

    def __init__(self, env, combos):
        super().__init__(env)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []

        for c in combos:
            arr = np.array([False] * env.action_space.n)

            for button in c:
                arr[buttons.index(button)] = True 
            
            self._decode_discrete_action.append(arr)
        
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))
    
    def action(self, action):
        return self._decode_discrete_action[action].copy()