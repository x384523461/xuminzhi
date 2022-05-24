from Solver import Solver
import  numpy as np
import matplotlib.pyplot as plt
from bernoullibandit import BernoulliBandit

class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon=epsilon
        self.estimates=np.array([init_prob]*self.bandit.k)
        # self.estimates=np.array(size=k)
    def run_one_step(self):
        if(np.random.random()<self.epsilon):
            k=np.random.randint(0,self.bandit.k)

        else:
            k=np.argmax(self.estimates)
        # print(k)
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


np.random.seed(1)
k=10
bandit_10_arm=BernoulliBandit(k)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


# np.random.seed(0)
# k=10
# bandit_10_arm=BernoulliBandit(k)
# epsilons=[1e-4,0.01,0.1,0.25,0.5]
# epsilon_greedy_solver_list=[EpsilonGreedy(bandit_10_arm,epsilon=e)for e in epsilons]
# epsilon_greedy_solver_names=["epsilon={}".format(e) for e in epsilons]
# for solver in epsilon_greedy_solver_list:
#     solver.run(5000)
# plot_results(epsilon_greedy_solver_list,epsilon_greedy_solver_names)


