from Solver import Solver
import numpy as np
import matplotlib.pyplot as plt
from bernoullibandit import BernoulliBandit
class DecayingEpsilonGreedy(Solver):
    def __init__(self,bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates=np.array([init_prob]*self.bandit.k)
        self.total_count=0
    def run_one_step(self):
        self.total_count+=1
        if(np.random.random()<1/self.total_count):
            k=np.random.randint(0,self.bandit.k)
        else:
            k=np.argmax(self.estimates)
        # print(self.total_count)
        r=self.bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])
        return k
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
# np.random.seed(1)    #试试这个随机
np.random.seed(0)
k=10
bandit_10_arm=BernoulliBandit(k)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon-衰减贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])