# %%
import torch
from torch import nn
from torch import distributions as dist
import numpy as np
# %%
class Lattice():
    def __init__(self, L, a=1., dim=2, g=1.):
        assert L%2 == 0
        self.L = L
        self.a = a
        self.d = dim
        self.g = g
        self.m = 1.
        
class Coupling(dist.transforms.Transform):
    """
    Defines a coupling layer for real nvp that transforms
    2 random variables to a variational distribution
    """
    def __init__(self, rev=False, depth=3, width=32, device='cpu'):
        super().__init__()
        assert depth > 1
        
        self.bijective = True
        self.domain = dist.constraints.real
        self.codomain = dist.constraints.real
        
        self.rev = rev
        
        s_layer = [nn.Linear(1, width),
                   nn.ReLU()]
        for _ in range(depth-2):
            s_layer.append(nn.Linear(width, width))
            s_layer.append(nn.ReLU())
        s_layer.append(nn.Linear(width, 1))
        self.s = nn.Sequential(*s_layer).to(device)
        
        t_layer = [nn.Linear(1, width),
                   nn.ReLU()]
        for _ in range(depth-2):
            t_layer.append(nn.Linear(width, width))
            t_layer.append(nn.ReLU())
        t_layer.append(nn.Linear(width, 1))
        self.t = nn.Sequential(*t_layer).to(device)
        
    def _call(self, x):
        def map(y1, y2):
            return self.t(y1) + y2*torch.exp(self.s(y1))
        
        y1 = x[:, 0:1]
        y2 = x[:, 1:2]
        if self.rev==False:
            y2 = map(y1, y2)
        else:
            y1 = map(y2, y1)
            
        return torch.cat([y1, y2], dim=1)
    
    def _inverse(self, x):
        def map(y1, y2):
            return torch.exp(-self.s(y1))*(y2 - self.t(y1))
        
        y1 = x[:, 0:1]
        y2 = x[:, 1:2]
        if self.rev==False:
            y2 = map(y1, y2)
        else:
            y1 = map(y2, y1)
        
        return torch.cat([y1, y2], dim=1)
    
    def log_abs_det_jacobian(self, x, y):
        if self.rev==False:
            return self.s(x[:, 0:1])
        else:
            return self.s(x[:, 1:2])

def RealNVP(nlayers=4, nn_depth=3, device='cpu'):
    assert (nlayers%2==0) and nlayers>1
    assert nn_depth > 1
    
    layers = []
    for _ in range(nlayers//2):
        layers.append(Coupling(False, nn_depth, device=device))
        layers.append(Coupling(True, nn_depth, device=device))      
    return layers
    
# %%
class DeepSets(nn.Module):
    def __init__(self, depth=3, width=32, agg="sum", 
                 nlayers=4, d2=3, device='cpu'):
        super().__init__()
        assert depth > 1
        
        layers1 = [nn.Linear(5, width),
                   nn.Tanh()]
        for _ in range(depth-1):
            layers1.append(nn.Linear(width, width))
            layers1.append(nn.ReLU())
        self.net1 = nn.Sequential(*layers1)
        
        if agg == "sum":
            self.aggregate = torch.sum
        elif agg == "logsumexp":
            self.aggregate = torch.logsumexp
        
        layers2 = []
        for _ in range(depth-1):
            layers2.append(nn.Linear(width, width))
            layers2.append(nn.ReLU())
        layers2.append(nn.Linear(width, 4))
        
        self.net2 = nn.Sequential(*layers2)
        self.rnvp = RealNVP(nlayers, d2, device)
        
    def forward(self, x):# (batch, 2*d-1, 5) array
        batch = x.shape[0]
        x2 = self.net1(x)
        # Aggregate along dim 1
        x3 = self.aggregate(x2, dim=1)# (batch, width) array
        # Obtain Gaussian parameters as output
        params = self.net2(x3)# (batch, 4) array
        mu, logsig = torch.split(params, 2, 
                                 dim=1) # (batch, 2)
        sig = torch.exp(logsig)# +ve standard deviation
        
        g = dist.Normal(mu, sig) # Gaussian of mu, sig
        p = dist.TransformedDistribution(g, self.rnvp)
        # Distribution after transformation by real nvp network
        
        x = p.rsample() # Gaussian mixture sample
        logp_x = p.log_prob(x) # Gaussian mixture log probability
        return x, logp_x

def sample(net: DeepSets, lattice: Lattice, batch, device='cpu'):
    L = lattice.L
    d = lattice.d
    
    # Lattice with 1 zero padding on every dim
    phi_pad = torch.zeros((batch,) + (L+1,)*2, device=device)
    phi_lp = torch.zeros([batch], device=device)
    # Log probabilities of samples
    
    for t in range(1, L+1):
        for x in range(1, L+1, 2):
            # Neighbour ϕ values at every lattice position
            with torch.no_grad(): # Ignore gradients of inputs
                phi_neigh = torch.stack(
                            [phi_pad[:, t, x-1], 
                            phi_pad[:, t-1, x],
                            phi_pad[:, t-1, x+1]], 1)
                # (batch, 2) array
            # Flags neighbours of x+1 against x
            flag_p = torch.zeros((batch, 3), device=device)
            flag_p[:, 2:] += 1
            # Flags if x or t = 1,2,L (batch, 3) arrays
            flag_1 = torch.stack(
                [torch.full((batch,), 1. if x==1 else 0., device=device),
                 torch.full((batch,), 1. if t==1 else 0., device=device),
                 torch.full((batch,), 1. if t==1 else 0., device=device)], 1)
            flag_2 = torch.stack(
                [torch.full((batch,), 1. if x==2 else 0., device=device),
                 torch.full((batch,), 1. if t==2 else 0., device=device),
                 torch.full((batch,), 1. if t==2 else 0., device=device)], 1)
            flag_L = torch.stack(
                [torch.full((batch,), 1. if x==L else 0., device=device),
                 torch.full((batch,), 1. if t==L else 0., device=device),
                 torch.full((batch,), 1. if t==L else 0., device=device)], 1)
            # Stacking them for NN input: (batch, 3, 5)
            inp = torch.stack([phi_neigh, flag_p, flag_1, 
                               flag_2, flag_L], 2)
            # Get sample and logprob from NN
            sample, logprob = net(inp)
            
            phi_pad[:, t, x:x+2] = sample
            phi_lp += logprob.sum(1)
            
    phi = phi_pad[:, 1:, 1:]
    return phi, phi_lp
# %%
def S_scalar(lattice: Lattice, phi, device='cpu'):
    """Computes scalar action for a lattice given lattice 
    object and a sample phi"""
    L = lattice.L
    d = lattice.d
    g = lattice.g
    m = lattice.m
    batch = phi.shape[0]
    
    # Compute d'Alembertian
    d_phi = torch.zeros((batch,), device=device)
    
    ind_0 = torch.arange(1, L-1, device=device)
    ind_l = torch.arange(L-2, device=device)
    ind_r = torch.arange(2, L, device=device)
    lat_dim = tuple(i for i in range(1,d+1))
    for j in lat_dim:
        phi_0 = torch.index_select(phi, j, ind_0)
        phi_left = torch.index_select(phi, j, ind_l)
        phi_right = torch.index_select(phi, j, ind_r)
        d_phi += torch.sum(phi_0*(2*phi_0 - phi_left - phi_right),
                           dim=lat_dim)
    d_phi *= L/(L-2)
    
    # Compute powers of phi
    phi_2 = (m**2)*torch.sum(phi**2, dim=lat_dim)
    phi_4 = g*torch.sum(phi**4, dim=lat_dim)
    
    # Compute action
    # S = d_phi + m**2*phi**2 + g*(phi**4)
    return d_phi + phi_2 + phi_4

def loss(lattice: Lattice, phi, phi_lp, device='cpu'):
    S = S_scalar(lattice, phi, device)
    
    return torch.mean(phi_lp + S)

def train(lattice: Lattice, net: DeepSets, batch=100, 
          epochs=600, lr=0.02, s_step=250, device='cpu'):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, s_step, 0.5)
    l_rec = torch.zeros((epochs//100, 100))
    
    for ep in range(epochs):
        # Zero your gradients for every batch
        optimizer.zero_grad()
        # Obtain samples and logpdf of given batch size
        phi, phi_lp = sample(net, lattice, batch, device)
        # Compute loss and gradients
        l = loss(lattice, phi, phi_lp, device)
        l.backward()
        # Adjust network parameters using optimizer and gradients
        optimizer.step()
        scheduler.step()
        
        l_rec[ep//100, ep%100] = l.item()
        if (ep+1)%100 == 0:
            print('loss_mean: {}'.format(l_rec[ep//100].mean()))
            print('loss_std: {}'.format(l_rec[ep//100].std()))

# %%
