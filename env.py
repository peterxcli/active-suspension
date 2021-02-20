import gym 
# import VL53L0X
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from env_obj import observation_space, action_space
import env_obj
import 

def make(envname):
    if envname == 'real':
        return real_env()
    else :
        return gym.make(envname)

class real_env(object):
    def __init__(self):
        self.observation_space = observation_space()
        self.action_space = action_space()

    def step(self, action):
        pass

    def reset(self):
        pass

if __name__ == "__main__":
    env = make('real')
    print(env.observation_space.shape[0])
    print(env.action_space.shape[0])
