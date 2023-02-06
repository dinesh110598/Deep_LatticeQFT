# %%
import torch
from torch import nn
# %%
class Lattice():
    def __init__(self, L, dim=2):
        self.L = L
        self.d = dim
        
    def init_phi(self):
        return torch.zeros((self.L,)*self.d)
    
    def dist_fun(self):
        dist = torch.zeros((self.d,) + (self.L,)*self.d)
        for k in range(self.d):
            for i in range(self.L):
                dist[k] += i / self.L
        return dist
    
class Net(nn.Module):
    def __init__(self, d, M, depth=3, width=16, agg="sum"):
        super().__init__()
        assert depth > 1
        
        layers1 = [nn.Linear(d, width),
                   nn.Tanh()]
        for _ in range(depth-1):
            layers1.append(nn.Linear(width, width))
            layers1.append(nn.ReLU())

        self.net1 = nn.Sequential(*layers1)
        
        self.aggregate = torch.sum
        
        layers2 = []
        for _ in range(depth-1):
            layers2.append(nn.Linear(width, width))
            layers2.append(nn.ReLU())
        layers2.append(nn.Linear(width, 3*M))
        
        self.net2 = nn.Sequential(*layers2)
        
    def forward(self, x):
        x2 = self.net1(x)
        x2 = self.aggregate(x2, dim=1)
        params = self.net2(x2)
        w, mu, logsig = torch.split(params, 3, dim=2)
        sig = torch.exp(0.5 * logsig)
        w = torch.softmax(w, dim=2)
        # Standard Gaussian random numbers
        rng = torch.randn_like(mu)
        
        return torch.sum(w*(mu + sig*rng), dim=2)
        
def sample(net, L, batch):
    phi_pad = torch.zeros((batch,) + (L+1,)*2).detach()
    
    for i in range(1, L+1):
        for j in range(1, L+1):
            phi_neigh = torch.stack(
                [phi_pad[:, i-1, j], phi_pad[:, i, j-1]], 1)
            # (batch, 2) array
            dist = torch.stack(
                [torch.full((batch,), (i-1)/L),
                 torch.full((batch,), (j-1)/L)], 1)
            # (batch, 2) array
            flag = torch.stack(
                [torch.full((batch,), 1 if i==1 else 0),
                 torch.full((batch,), 1 if j==1 else 0)], 1)
            # (batch, 2) array
            
            inp = torch.stack([phi_neigh, dist, flag], 2)
            phi_pad[:, i, j] = net(inp)
            
