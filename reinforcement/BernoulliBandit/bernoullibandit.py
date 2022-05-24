import numpy as np
import matplotlib.pyplot as plt
class BernoulliBandit:
    def __init__(self,k):
        self.probs=np.random.uniform(size=k)
        self.best_idx=np.argmax(self.probs)
        self.best_prob=self.probs[self.best_idx]
        self.k=k
    def step(self,k):
        if np.random.rand()<self.probs[k]:
            return 1
        else:
            return 0
# np.random.seed(1)
# k=10
# bandit_10_arm=BernoulliBandit(k)
# print("随机生成了一个%d臂伯努利老虎机" % k)
# print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
#       (bandit_10_arm.best_idx, bandit_10_arm.best_prob))