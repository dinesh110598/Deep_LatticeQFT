# %%
import torch
from torch import nn
from torch import distributions as dist
from torch.nn import functional as F
import numpy as np
# %%
class Lattice:
    def __init__(self, L, g, m_2=-4):
        self.L = L
        self.m_2 = m_2
        self.g = g
        
    def S(self, phi):
        m_2 = self.m_2
        g = self.g
        
        phi_2 = m_2*torch.sum(phi**2, dim=(1,2))
        phi_4 = g*torch.sum(phi**4, dim=(1,2))
        
        d_phi = torch.sum(phi[:,1:-1,:]*(2*phi[:,1:-1,:] - 
                          phi[:,:-2,:] - phi[:,2:,:]), dim=(1,2))
        d_phi += torch.sum(phi[:,:,1:-1]*(2*phi[:,:,1:-1] -
                           phi[:,:,:-2] - phi[:,:,2:]), dim=(1,2))
        
        return d_phi + phi_2 + phi_4
    
    def KL_loss(self, phi, phi_lp):
        S = self.S(phi)
        return phi_lp + S
    
    def pos_x_t(self, batch, l):
        L = self.L

        pos_t = torch.arange(L).unsqueeze(1)/L
        pos_t = pos_t.expand(batch, L, L).float()
        pos_t = torch.cat([torch.zeros([batch, 1, L]), pos_t], dim=1)
        pos_t = torch.cat([torch.zeros([batch, L+1, l]),
                           pos_t,
                           torch.zeros([batch, L+1, l])], dim=2)

        pos_x = torch.arange(L).unsqueeze(0)/L
        pos_x = pos_x.expand(batch, L, L).float()
        pos_x = torch.cat([torch.zeros([batch, 1, L]), pos_x], dim=1)
        pos_x = torch.cat([torch.zeros([batch, L+1, l]),
                           pos_x,
                           torch.zeros([batch, L+1, l])], dim=2)
        return pos_t, pos_x

# 2-var RealNVP network
def backbone(depth=4, width=32, s_net=True):
    layers = [nn.Conv1d(3, width, 1),
              nn.LeakyReLU()]
    for _ in range(depth-2):
        layers += [nn.Conv1d(width, width, 1),
                   nn.LeakyReLU()]
    layers += [nn.Conv1d(width, 1, 1)]
    if s_net:
        layers += [nn.Tanh()]
    return nn.Sequential(*layers)

class RealNVP(nn.Module):
    def __init__(self, ncouplings=4, depth=4, width=32):
        super().__init__()
        self.ncouplings = ncouplings
        self.s = nn.ModuleList([backbone(depth, width, True) 
                                for _ in range(ncouplings)])
        self.t = nn.ModuleList([backbone(depth, width, False) 
                                for _ in range(ncouplings)])
        
        self.s_scale = nn.ParameterList([
            nn.Parameter(torch.randn([])) 
            for _ in range(ncouplings)])
    # Conditional coupling call
    def forward(self, x, theta):
        # Assume x is sampled from random Gaussians
        s_vals = []
        y1, y2 = x[:, :, :1], x[:, :, 1:]
        t1, t2 = theta[:, :, :1], theta[:, :, 1:]
        for i in range(self.ncouplings):
            if i%2 == 0:
                x1, x2 = y1, y2
                y1 = x1
                x1 = torch.cat([x1, t1], dim=1)
                s = self.s_scale[i]*self.s[i](x1)
                y2 = torch.exp(s)*x2 + self.t[i](x1)
            else:
                x1, x2 = y1, y2
                y2 = x2
                x2 = torch.cat([x2, t2], dim=1)
                s = self.s_scale[i]*self.s[i](x2)
                y1 = torch.exp(s)*x1 + self.t[i](x2)
            s_vals.append(s)
        
        return torch.cat([y1, y2], 2), torch.cat(s_vals, 2).sum(2)
    
    def inv(self, x, theta):
        # x here is a sample of the latent distribution
        s_vals = []
        y1, y2 = x[:, :, :1], x[:, :, 1:]
        t1, t2 = theta[:, :, :1], theta[:, :, 1:]
        for i in reversed(range(self.ncouplings)):
            if i%2 == 0:
                x1, x2 = y1, y2
                y1 = x1
                x1 = torch.cat([x1, t1], dim=1)
                s = self.s_scale[i]*self.s[i](x1)
                y2 = (x2 - self.t[i](x1))*torch.exp(-s)
            else:
                x1, x2 = y1, y2
                y2 = x2
                x2 = torch.cat([x2, t2], dim=1)
                s = self.s_scale[i]*self.s[i](x2)
                y1 = (x1 - self.t[i](x2))*torch.exp(-s)
            s_vals.append(-s)
        return torch.cat([y1, y2], 2), torch.cat(s_vals, 2).sum(2)
# %%
def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel=3,
                 skip=False):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels//2, kernel)
        self.bn = nn.BatchNorm1d(in_channels//2)
        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.relu = nn.ReLU()
        if skip:
            self.skip_net = nn.Conv1d(in_channels, in_channels//2, 1)
        self.conv2 = nn.Conv1d(in_channels, 2*out_channels, kernel)
        
    def forward(self, x, a=None, last=False):
        y = self.conv(x)
        y = self.bn(y)
        if a is not None:
            s1 = (y.shape[-1]-1)//2
            s2 = (a.shape[-1]-1)//2
            y += self.skip_net(a[:, :, s2-s1:s1-s2])
        y = concat_elu(y)
        
        y = self.relu(self.conv1(y))
        
        y = self.conv2(y)
        a, b = torch.chunk(y, 2, dim=1)
        c = a * torch.sigmoid(b)
        if last:
            return c
        else:
            return c + x[:, :, 2:-2]

class GatedConvChain(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3,
                 skip=False):
        super(GatedConvChain, self).__init__()
        self.chain = nn.ModuleList([GatedConv(in_channels, 
                                              out_channels, kernel, skip)])
        
    def forward(self, x, a=None):
        y = self.chain[0](x, a)
        for i in range(1, 2):
            y = self.chain[i](y)
        return y
    
class ConvNet(nn.Module):
    # Improve the initialization of weights?
    def __init__(self, width=32):
        super(ConvNet, self).__init__()
        self.l = 11
        
        init_layer = [nn.Conv1d(4, width, 2),
                      nn.ELU(),
                      nn.Conv1d(width, width, 3),
                      nn.ELU()]
        nets = nn.ModuleList([nn.Sequential(*init_layer)])
        nets.append(GatedConv(width, width))
        
        nets.append(GatedConv(width, width))        
        nets.append(GatedConv(width, width))
        
        nets.append(GatedConv(width, width, skip=True))    
        nets.append(GatedConv(width, 4, skip=True))
        
        self.realnvp = RealNVP(6, 4)
        self.nets = nets
    
    def forward(self, x):
        batch = x.shape[0]
        
        stream = [self.nets[0](x)]
        stream += [self.nets[1](stream[-1])]
        
        stream += [self.nets[2](stream[-1])]
        stream += [self.nets[3](stream[-1])]
        
        stream += [self.nets[4](stream[-1], stream[1])]
        stream += [self.nets[5](stream[-1], stream[0], True)]
        
        out = stream[-1].permute(0, 2, 1)
        mu, sig = torch.chunk(out, 2, dim=2)
        sig = sig.exp() + 1e-6
        gauss = dist.Independent(dist.Normal(mu, sig), 1)
        gauss_x = gauss.rsample()
        gauss_lp = gauss.log_prob(gauss_x)
        
        theta = torch.cat([mu, sig], dim=1)
        z, logdetJ = self.realnvp(gauss_x, theta)
        
        z = z[:, 0, :]
        logp_z = (gauss_lp - logdetJ)[:, 0]
        return z, logp_z
    
    def logprob(self, z, x):
        batch = x.shape[0]
        
        stream = [self.nets[0](x)]
        stream += [self.nets[1](stream[-1])]
        
        stream += [self.nets[2](stream[-1])]
        stream += [self.nets[3](stream[-1])]
        
        stream += [self.nets[4](stream[-1], stream[1])]
        stream += [self.nets[5](stream[-1], stream[0], True)]
        
        out = stream[-1].permute(0, 2, 1)
        mu, sig = torch.chunk(out, 2, dim=2)
        sig = sig.exp() + 1e-6
        gauss = dist.Independent(dist.Normal(mu, sig), 1)
        
        theta = torch.cat([mu, sig], dim=1)
        z = z.unsqueeze(1)
        gauss_x, logdetJ_inv = self.realnvp.inv(z, theta)
        gauss_lp = gauss.log_prob(gauss_x)
        logp_z = (gauss_lp + logdetJ_inv)[:, 0]
        return logp_z
        
# %%
def sample_fn(lattice: Lattice, net: ConvNet,
              batch=100, device='cpu'):
    L = lattice.L
    l = net.l

    phi_pad = torch.zeros((batch, L+1, L+2*l), device=device)
    phi_flag = torch.ones((batch, L+1, L+2*l), device=device)
    phi_flag[:, 1:L, l:L+l] = 0
    phi_lp = torch.zeros((batch,), device=device)
    pos_t, pos_x = lattice.pos_x_t(batch, l)
    pos_t = pos_t.to(device)
    pos_x = pos_x.to(device)

    for t in range(1, L+1):
        for x in range(l, l+L, 2):
            with torch.no_grad():
                dep_set = torch.cat([phi_pad[:, t:t+1, x-l:x],
                                    phi_pad[:, t-1:t, x:x+l+2]],
                                    dim=2)
            # dep_set.shape = (batch, 1, 2*l + 1)
            dep_flag = torch.cat([phi_flag[:, t:t+1, x-l:x],
                                 phi_flag[:, t-1:t, x:x+l+2]],
                                 dim=2)
            dep_t = torch.cat([pos_t[:, t:t+1, x-l:x],
                               pos_t[:, t-1:t, x:x+l+2]],
                              dim=2)
            dep_x = torch.cat([pos_x[:, t:t+1, x-l:x],
                               pos_x[:, t-1:t, x:x+l+2]],
                              dim=2)
            
            dep_set = torch.cat([dep_set, dep_flag,
                                 dep_t, dep_x], dim=1)
            
            sample, logp = net(dep_set)
            phi_pad[:, t, x:x+2] = sample
            phi_lp += logp
    phi = phi_pad[:, 1:L+1, l:l+L]
    return phi, phi_lp

def logprob_fn(lattice: Lattice, net: ConvNet, phi, 
            device='cpu'):
    
    L = lattice.L
    l = net.l
    batch = phi.shape[0]

    phi_pad = torch.cat([torch.zeros([batch, 1, L], device=device), phi], dim=1)
    phi_pad = torch.cat([torch.zeros([batch, L+1, l], device=device),
                         phi_pad,
                         torch.zeros([batch, L+1, l], device=device)], dim=2)
    
    phi_flag = torch.ones((batch, L+1, L+2*l), device=device)
    phi_flag[:, 1:L, l:L+l] = 0
    
    phi_lp = torch.zeros((batch,), device=device)
    pos_t, pos_x = lattice.pos_x_t(batch, l)
    pos_t = pos_t.to(device)
    pos_x = pos_x.to(device)
    
    for t in range(1, L+1):
        for x in range(l, l+L, 2):
            with torch.no_grad():
                dep_set = torch.cat([phi_pad[:, t:t+1, x-l:x],
                                    phi_pad[:, t-1:t, x:x+l+2]],
                                    dim=2)
            # dep_set.shape = (batch, 1, 2*l + 1)
            dep_flag = torch.cat([phi_flag[:, t:t+1, x-l:x],
                                 phi_flag[:, t-1:t, x:x+l+2]],
                                 dim=2)
            dep_t = torch.cat([pos_t[:, t:t+1, x-l:x],
                               pos_t[:, t-1:t, x:x+l+2]],
                              dim=2)
            dep_x = torch.cat([pos_x[:, t:t+1, x-l:x],
                               pos_x[:, t-1:t, x:x+l+2]],
                              dim=2)
            
            dep_set = torch.cat([dep_set, dep_flag,
                                 dep_t, dep_x], dim=1)

            logp = net.logprob(phi_pad[:, t, x:x+2], dep_set)
            phi_lp += logp
    
    return phi_lp

def loss_symmetric(lattice: Lattice, net: ConvNet, batch=100,
                     device='cpu'):
    
    phi, phi_lp = sample_fn(lattice, net, batch, device)
    S_scalar = lattice.S(phi)
    phi_tot = phi
    for k in range(1,4):
        phi_rot = torch.rot90(phi, k, [1,2]).detach()
        phi_lp1 = logprob_fn(lattice, net, phi_rot, device)
        phi_lp += phi_lp1
        
    for rdim in [1,2]:
        phi_ref = torch.flip(phi, dims=[rdim]).detach()
        phi_lp1 = logprob_fn(lattice, net, phi_ref, device)
        phi_lp += phi_lp1
    
    return (phi_lp/6 + S_scalar).mean()
# %%
def train(lattice: Lattice, net: ConvNet, batch=100,
          epochs=100, lr=0.01, schedule_int=400, device='cpu'):
    optimizer = torch.optim.Adam(net.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_int, 
                                                0.5)

    for ep in range(epochs):
        # Zero your gradients for every batch
        optimizer.zero_grad()
        # Compute loss and gradients
        l = loss_symmetric(lattice, net, batch, device)
        l.backward()
        # Adjust network parameters using optimizer and gradients
        optimizer.step()
        scheduler.step()

        if (ep+1) % 200 == 0:
            loss = lattice.KL_loss(phi, phi_lp)
            print('loss_mean: {}'.format(loss.mean()))
            print('loss_std: {}'.format(loss.std()))

# %%
# lattice = Lattice(16, 5.05)
# device = 'cpu'
# net = ConvNet(64).to(device)
# # %%
# epochs = 1
# batch = 50
# train(lattice, net, batch, epochs, 2e-5, device)
# %%
# torch.save(net.state_dict(), 'ar_flow_2d_chkpt.pth')
# %%
net.load_state_dict(torch.load('Saves/ar_flow_2d_chkpt.pth'))
# %%