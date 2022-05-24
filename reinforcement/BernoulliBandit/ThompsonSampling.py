from Solver import Solver
import numpy as np
import matplotlib.pyplot as plt
from bernoullibandit import BernoulliBandit
class ThompsonSampling(Solver):
    def __init__(self,bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a=np.ones(self.bandit.k)
        self._b=np.ones(self.bandit.k)
    def run_one_step(self):
        samples=np.random.beta(self._a,self._b)
        k=np.argmax(samples)
        r=self.bandit.step(k)
        self._a[k]+=r
        self._b[k]+=(1-r)
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
np.random.seed(0)
k=10
bandit_10_arm=BernoulliBandit(k)
Thompson_epsilon_greedy_solver = ThompsonSampling(bandit_10_arm)
Thompson_epsilon_greedy_solver.run(5000)
print('汤姆森采样算法的累积懊悔为：', Thompson_epsilon_greedy_solver.regret)
plot_results([Thompson_epsilon_greedy_solver], ["ThompsonSampling"])