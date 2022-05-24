from Solver import Solver
import numpy as np
import matplotlib.pyplot as plt
from bernoullibandit import BernoulliBandit
class UCB(Solver):
    def __init__(self,bandit,coef,init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count=0
        self.estimates=np.array([init_prob]*10)
        self.coef=coef
    def run_one_step(self):
        self.total_count+=1
        ucb=self.estimates+self.coef*np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))
        k=np.argmax(ucb)
        r=self.bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])
        return  k
def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
















        plt.plot(time_list, solver.regrets, label=solver_names[idx])

    plt.xlabel('Time steps')

    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.k)
    plt.legend()
    plt.show()
np.random.seed(1)
k=10
coef=1
bandit_10_arm=BernoulliBandit(k)
decaying_epsilon_greedy_solver = UCB(bandit_10_arm,coef)
decaying_epsilon_greedy_solver.run(5000)
print('上置信界算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["UCB"])
