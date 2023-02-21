# %%
import torch
from torch import nn
from torch import distributions as dist
import numpy as np
# %%
class Ising:
    def __init__(self, L, beta, J=-1):
        self.L = L
        self.beta = beta
        self.J = J
        
    def energy(self, x):
        assert self.L == x.shape[1]
        
        e = (torch.sum(x[:, :-1, :]*x[:, 1:, :], dim=(1,2)) 
             + torch.sum(x[:, :, :-1]*x[:, :, 1:], dim=(1,2)))
        return self.J * e
    
    def KL_loss(self, phi, phi_lp, beta):
        return phi_lp + beta*self.energy(phi)
    
class ConvNet(nn.Module):
    # Improve the initialization of weights?
    def __init__(self, l, filter_w=2, depth=2, width=32):
        super(ConvNet, self).__init__()
        assert l>1
        assert (l-1)%filter_w == 0
        assert depth > 1
        self.l = l
        self.fw = filter_w
        kw = filter_w*2 + 1
        
        init_layer = [nn.Conv1d(1, width, 3),
                      nn.LeakyReLU(0.1)]
        for j in range(depth-1):
            init_layer.append(nn.Conv1d(width, width, 1))
            init_layer.append(nn.LeakyReLU(0.1))
        nets = nn.ModuleList([nn.Sequential(*init_layer)])
        
        for i in range((l-1)//filter_w - 1):
            mid_layer = [nn.Conv1d(width, width, kw),
                    nn.LeakyReLU(0.1)]
            for j in range(depth-1):
                mid_layer.append(nn.Conv1d(width, width, 1))
                mid_layer.append(nn.LeakyReLU(0.1))
            nets.append(nn.Sequential(*mid_layer))
            
        end_layer = [nn.Conv1d(width, width, kw),
                    nn.LeakyReLU(0.1)]
        for j in range(depth-2):
            end_layer.append(nn.Conv1d(width, width, 1))
            end_layer.append(nn.LeakyReLU(0.1))
        end_layer.append(nn.Conv1d(width, 2, 1))
        nets.append(nn.Sequential(*end_layer))
        self.nets = nets
        
    # Make residual connection operator expsumlog instead of 1
    def forward(self, x):
        batch = x.shape[0]
        
        x1 = self.nets[0](x)
        for i in range(1, (self.l-1)//self.fw):
            x2 = self.nets[i](x1)
            x2 += x1[:, :, self.fw:-self.fw]
            x1 = x2
        x2 = self.nets[-1](x1)[:, :, 0]
        # Categorical distribution with logits x2
        d = dist.Categorical(logits=x2)
        # Sample and log prob of d
        sample = d.sample()
        logp = d.log_prob(sample)
        sample = sample*2 - 1 # To get -1,1 values
        return sample, logp
    
def sample_fn(lattice: Ising, net: ConvNet, 
              batch=100, device='cpu'):
    L = lattice.L
    l = net.l
    
    phi_pad = torch.zeros((batch, L+1 , L+2*l), device=device)
    phi_lp = torch.zeros((batch,), device=device)
    
    for y in range(1, L+1):
        for x in range(l, l+L):
            dep_set = torch.cat([phi_pad[:, y:y+1, x-l:x],
                                 phi_pad[:, y-1:y, x:x+l+1]],
                                dim=2)
            # dep_set.shape = (batch, 1, 2*l + 1)
            if (y==1) and (x==l):
                sample = torch.randint(2, (batch,), device=device)*2. - 1
                logp = torch.full((batch,), -np.log(2), device=device)
            else:
                sample, logp = net(dep_set)
            phi_pad[:, y, x] = sample
            phi_lp += logp
    phi = phi_pad[:, 1:L+1, l:l+L]
    return phi, phi_lp
# %%
def loss_reinforce(lattice: Ising, phi, phi_lp, beta):
    with torch.no_grad():
        loss = lattice.KL_loss(phi, phi_lp, beta)
        loss = loss - loss.mean()
        
    return torch.mean(phi_lp*loss)

def train(lattice: Ising, net: ConvNet, batch=100, 
          epochs=100, lr=0.02, device='cpu', anneal=True):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, 0.5)
    
    for ep in range(epochs):
        # Zero your gradients for every batch
        optimizer.zero_grad()
        # Obtain samples and logp of given batch size
        phi, phi_lp = sample_fn(lattice, net, batch, device)
        # Compute loss and gradients
        if anneal:
            beta_conv = lattice.beta
            beta = beta_conv*(1 - 0.95**ep)
        else:
            beta = lattice.beta
        l = loss_reinforce(lattice, phi, phi_lp, beta)
        l.backward()
        # Adjust network parameters using optimizer and gradients
        optimizer.step()
        scheduler.step()
        
        if (ep+1)%200 == 0:
            loss = lattice.KL_loss(phi, phi_lp, beta)
            print('loss_mean: {}'.format(loss.mean()))
            print('loss_std: {}'.format(loss.std()))
# %%
def metropolis(lattice: Ising, net: ConvNet, batch=500, 
               steps=10000, device='cpu'):
    epochs = steps//batch
    acc = 0
    curr_x = None
    curr_model_lp = None
    curr_target_lp = None
    
    for ep in range(epochs):
        with torch.no_grad():
            phi, phi_lp = sample_fn(lattice, net, batch, device)
        S = lattice.beta*lattice.energy(phi)
        if ep==0:
            curr_x = phi[0]
            curr_model_lp = phi_lp[0]
            curr_target_lp = -S[0]
            start = 1
        else:
            start = 0
        
        for i in range(start, batch):
            prop_x = phi[i]
            prop_model_lp = phi_lp[i]
            prop_target_lp = -S[i]
            acc_prob = (curr_model_lp - prop_model_lp +
                        prop_target_lp - curr_target_lp).exp().item()
            if np.random.rand() < acc_prob:
                curr_x = prop_x
                curr_model_lp = prop_model_lp
                curr_target_lp = prop_target_lp
                acc += 1
                
    return acc
# %%