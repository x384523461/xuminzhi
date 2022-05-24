import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm   # 是显示循环进度条的库

class CliffWalkingEnv:
    def __init__(self,ncol,nrow):
        self.nrow=nrow
        self.ncol=ncol
        self.x=0                 #智联体当前的位置
        self.y=self.nrow-1       #智联体当前的位置

    def step(self,action):
        change=[[0,-1],[0,1],[-1,0],[1,0]]
        self.x=min(self.ncol-1,max(0,self.x+change[action][0])) #自己保存自己当前的位置，不需要重新传入参数
        self.y=min(self.nrow-1,max(0,self.y+change[action][1]))
        next_state=self.y*self.ncol+self.x
        reward=-1
        done=False
        if(self.y==self.nrow-1 and self.x>0):
            done=True
            if(self.x!=self.ncol-1):
                reward=-100
        return next_state,reward,done
    def reset(self):
        self.x=0
        self.y=self.nrow-1
        return self.y*self.ncol+self.x        #返回起始点的s



