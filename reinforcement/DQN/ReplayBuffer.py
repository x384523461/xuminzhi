import random
import gym
import numpy as np
import collections
from tqdm import tqdm


import matplotlib.pyplot as plt
import rl_utils
class ReplayBuffer:                        #经验回放池
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
    def sample(self,batch_size):
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done
    def size(self):
        return len(self.buffer)
