import torch
import torch.nn.functional as F
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=torch.nnn.Linear(state_dim,hidden_dim)
        self.fc2=torch.nn.Linear(hidden_dim,action_dim)
    def forward(self,x):
        x=F.relu(self.fc1(1))
        return self.fc2(x)


