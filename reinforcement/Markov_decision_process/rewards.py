import numpy as np
np.random.seed(0)
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
def compute_return(start_index,chain,gamma):
    G=0
    for i in reversed(range(start_index,len(chain))):
        G=gamma*G+rewards[chain[i]-1]
    return G
chain=[1,2,3,6]
start_index=0
G=compute_return(start_index,chain,gamma)
print("根据本序列的计算得到的回报：%s"%G)
