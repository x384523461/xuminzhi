from CliffWalkingEnc import CliffWalkingEnc
class ValueIteration:
    def __init__(self,env,theta,gamma):
        self.env=env
        self.v=[0]*self.env.ncol*self.env.nrow
        self.theta=theta
        self.gamma=gamma
        self.pi=[None for i in range(self.env.ncol*self.env.nrow)]
    def value_iteration(self):
        cnt=0
        while 1:
            max_diff=0
            new_v=[0]*self.env.ncol*self.env.nrow
            for s in range(self.env.nrow*self.env.ncol):
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    for res in self.env.P[s][a]:
                        p,next_state,r,done=res
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                    qsa_list.append(qsa)
                new_v[s]=max(qsa_list)
                max_diff=max(max_diff,abs(new_v[s]-self.v[s]))
            self.v=new_v
            # self.get_policy()
            if(max_diff<self.theta):
                break
            cnt+=1
        print("价值迭代一共进行%d轮"%cnt)
        self.get_policy()
    def get_policy(self):
        for s in range(self.env.ncol*self.env.nrow):
            qsa_list=[]
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p,next_state,r,done=res
                    qsa+=r+p*self.gamma*self.v[next_state]*(1-done)
                qsa_list.append(qsa)
            maxq=max(qsa_list)
            cntq=qsa_list.count(maxq)
            self.pi[s]=[1/cntq if q==maxq else 0for q in qsa_list]


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()
env=CliffWalkingEnc()
action_meaning=['^', 'v', '<', '>']
theta=0.001
gamma=0.9
agent=ValueIteration(env,theta,gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])

