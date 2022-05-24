import numpy as np

def compute(p,rewards,gamma,states_num):
    rewards=np.array(rewards).reshape((-1,1))
    value=np.dot(np.linalg.inv(np.eye(states_num,states_num)-gamma*p),rewards)
    return value
p=[
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
p=np.array(p)
rewards=[-1,-2,-2,10,1,0]
gamma=0.5
v=compute(p,rewards,gamma,6)
print("mrp中每一个状态价值分别为\n",v)