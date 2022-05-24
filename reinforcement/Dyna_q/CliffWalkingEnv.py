import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time
class CliffWalkingEnv:
    def __init__(self,ncol,nrow):
        self.ncol=ncol
        self.nrow=nrow
        self.x=0
        self.y=self.nrow-1  #智联体初始的位置
    def step(self,action):
        change=[[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x=min(self.ncol-1,max(0,self.x+change[action][0]))
        self.y=min(self.nrow-1,max(0,self.y+change[action][1]))
        next_state=self.y*self.ncol+self.x
        reward=-1
        done=False
        if(self.y==self.nrow-1 and self.x>0):
            done=True
            if(self.x!=self.ncol-1):        #下一个位置为悬崖
                reward=-100
        return next_state,reward,done
    def reset(self):
        self.x=0
        self.y=self.nrow-1
        return self.y*self.ncol+self.x
