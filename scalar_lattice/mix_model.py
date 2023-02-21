# %%
import torch
from torch import nn
from torch import distributions as dist
import numpy as np
# %%
class Lattice():
    def __init__(self, L, a=1., dim=2, g=1.):
        self.L = L
        self.a = a
        self.d = dim
        self.g = g
        self.m = 1.


class DeepSets(nn.Module):
    def __init__(self, depth=3, width=32, agg="sum", M=4):
        super().__init__()
        assert depth > 1
        
        layers1 = [nn.Linear(4, width),
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
        layers2.append(nn.Linear(width, 3*M))
        
        self.net2 = nn.Sequential(*layers2)
        
    def forward(self, x):# (batch, d, 4) array
        batch = x.shape[0]
        x2 = self.net1(x)
        # Aggregate along dim 1
        x3 = self.aggregate(x2, dim=1)# (batch, width) array
        # Obtain Gaussian parameters as output
        params = self.net2(x3)# (batch, 3*M) array
        w, mu, logsig = torch.split(params, params.shape[1]//3, 
                                    dim=1) # (batch, M)
        sig = torch.exp(logsig)# +ve standard deviation
        w = nn.functional.softmax(w, 1)# +ve and add upto 1
        
        c = dist.Categorical(w) # Categorical dist of w
        g = dist.Normal(mu, sig) # Gaussian of mu, sig
        gmm = dist.MixtureSameFamily(c, g) # Mixed gaussian
        
        x = gmm.sample() # Gaussian mixture sample
        logp_x = gmm.log_prob(x) # Gaussian mixture log probability
        return x, logp_x
        
def sample(net: DeepSets, lattice: Lattice, batch, device='cpu'):
    L = lattice.L
    d = lattice.d
    
    # Lattice with 1 zero padding on every dim
    phi_pad = torch.zeros((batch,) + (L+1,)*2, device=device)
    phi_lp = torch.zeros([batch], device=device)
    # Log probabilities of samples
    
    for t in range(1, L+1):
        for x in range(1, L+1):
            # Neighbour ϕ values at every lattice position
            with torch.no_grad(): # Ignore gradients of inputs
                phi_neigh = torch.stack(
                            [phi_pad[:, t, x-1], 
                            phi_pad[:, t-1, x]], 1)
                # (batch, 2) array
            # Flags if x or t = 1,2,L (batch, 2) arrays
            flag_1 = torch.stack(
                [torch.full((batch,), 1. if x==1 else 0., device=device),
                 torch.full((batch,), 1. if t==1 else 0., device=device)], 1)
            flag_2 = torch.stack(
                [torch.full((batch,), 1. if x==2 else 0., device=device),
                 torch.full((batch,), 1. if t==2 else 0., device=device)], 1)
            flag_L = torch.stack(
                [torch.full((batch,), 1. if x==L else 0., device=device),
                 torch.full((batch,), 1. if t==L else 0., device=device)], 1)
            # Stacking them for NN input: (batch, 2, 4)
            inp = torch.stack([phi_neigh, flag_1, 
                               flag_2, flag_L], 2)
            # Get sample and logprob from NN
            sample, logprob = net(inp)
            
            phi_pad[:, t, x] = sample
            phi_lp += logprob
            
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

def loss_fn(lattice, phi, phi_lp, device):
    S = S_scalar(lattice, phi, device)
    batch = phi.shape[0]

    return phi_lp + S

def loss_reinforce(lattice: Lattice, phi, phi_lp, device='cpu'):
    with torch.no_grad():
        loss = loss_fn(lattice, phi, phi_lp, device)
        loss = loss - loss.mean()
    
    return torch.mean(phi_lp*loss)

def train(lattice: Lattice, net: DeepSets, batch=100, 
          epochs=600, device='cpu'):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.6)
    rec_size = np.min(np.asarray([epochs, 100]))
    l_rec = torch.zeros((epochs//rec_size, rec_size))
    
    for ep in range(epochs):
        # Zero your gradients for every batch
        optimizer.zero_grad()
        # Obtain samples and logpdf of given batch size
        phi, phi_lp = sample(net, lattice, batch, device)
        # Compute loss and gradients
        l = loss_reinforce(lattice, phi, phi_lp, device)
        l.backward()
        # Adjust network parameters using optimizer and gradients
        optimizer.step()
        scheduler.step()
        
        l_rec[ep//rec_size, ep%rec_size] = l.item()
        if (ep+1)%rec_size == 0:
            print('loss_mean: {}'.format(l_rec[ep//rec_size].mean()))
            print('loss_std: {}'.format(l_rec[ep//rec_size].std()))
# %%
lattice = Lattice(25)
device = 'cuda:0'
net = DeepSets().to(device)
batch = 100
train(lattice, net, batch, 600, device)
# %%
phi, phi_lp = sample(net, lattice, batch, device)
ls = loss_fn(lattice, phi, phi_lp, device)
ls.std()
# %%
