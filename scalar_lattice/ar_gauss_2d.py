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

        phi_2 = m_2*torch.sum(phi**2, dim=(1, 2))
        phi_4 = g*torch.sum(phi**4, dim=(1, 2))

        d_phi = torch.sum(phi[:, 1:-1, :]*(2*phi[:, 1:-1, :] -
                          phi[:, :-2, :] - phi[:, 2:, :]), dim=(1, 2))
        d_phi += torch.sum(phi[:, :, 1:-1]*(2*phi[:, :, 1:-1] -
                           phi[:, :, :-2] - phi[:, :, 2:]), dim=(1, 2))

        return d_phi + phi_2 + phi_4

    def KL_loss(self, phi, phi_lp):
        S = self.S(phi)
        return phi_lp + S

    def pos_x_t(self, batch, l):
        L = self.L

        pos_t = torch.arange(L).unsqueeze(1)/L
        pos_t = torch.broadcast_to(pos_t, [batch, L, L])
        pos_t = torch.cat([torch.zeros([batch, 1, L]), pos_t], dim=1)
        pos_t = torch.cat([torch.zeros([batch, L+1, l]),
                           pos_t,
                           torch.zeros([batch, L+1, l])], dim=2)

        pos_x = torch.arange(L).unsqueeze(0)/L
        pos_x = torch.broadcast_to(pos_x, [batch, L, L])
        pos_x = torch.cat([torch.zeros([batch, 1, L]), pos_x], dim=1)
        pos_x = torch.cat([torch.zeros([batch, L+1, l]),
                           pos_x,
                           torch.zeros([batch, L+1, l])], dim=2)
        return pos_t, pos_x


def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels=32, kernel=3,
                 skip=False):
        super(GatedConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels//2, kernel)
        self.bn = nn.BatchNorm1d(in_channels//2)
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
        for i in range(1):
            self.chain += [GatedConv(in_channels, out_channels)]

    def forward(self, x, a=None):
        y = self.chain[0](x, a)
        for i in range(1, 2):
            y = self.chain[i](y)
        return y


class ConvNet(nn.Module):
    # Improve the initialization of weights?
    def __init__(self, M=6, width=32):
        super(ConvNet, self).__init__()
        self.l = 12
        self.M = M

        init_layer = [nn.Conv1d(4, width, 3),
                      nn.ELU()]
        nets = nn.ModuleList([nn.Sequential(*init_layer)])

        nets.append(GatedConv(width, width))
        nets.append(nn.Sequential(
            nn.Conv1d(width, width, 3, 2),
            nn.ELU()))
        nets.append(GatedConv(width, width, skip=True))

        nets.append(GatedConv(width, 3*M, skip=True))
        self.nets = nets

    def forward(self, x, device='cpu'):
        batch = x.shape[0]

        stream = [self.nets[0](x)]
        stream += [self.nets[1](stream[-1])]
        stream += [self.nets[2](stream[-1])]
        stream += [self.nets[3](stream[-1], stream[1])]
        stream += [self.nets[4](stream[-1], stream[0], True)]

        y = stream[-1][:, :, 0]
        w, mu, sig = torch.chunk(y, 3, dim=1)
        logw = nn.functional.log_softmax(w, 1)
        sig = sig.exp() + 1e-6
        cat = dist.Categorical(logits=logw)
        gauss = dist.Normal(mu, sig)
        s_cat = cat.sample()
        s_gauss = gauss.sample()
        # Mixture of gaussians sample
        sample = s_gauss[torch.arange(batch), s_cat]
        sample2 = sample.unsqueeze(1).broadcast_to((batch, self.M))
        # Gauss logprob for every sample
        gauss_lp = gauss.log_prob(sample2)
        # Mixed Gauss logprob
        mixed_lp = torch.logsumexp(logw + gauss_lp, dim=1)

        return sample, mixed_lp


def sample_fn(lattice: Lattice, net: ConvNet,
              batch=100, device='cpu'):
    L = lattice.L
    l = net.l

    phi_pad = torch.zeros((batch, L+1, L+2*l), device=device)
    phi_flag = torch.ones((batch, L+1, L+2*l), device=device)
    phi_flag[:, 1:L+1, l:L+l] = 0
    phi_lp = torch.zeros((batch,), device=device)
    pos_t, pos_x = lattice.pos_x_t(batch, l)
    pos_t = pos_t.to(device)
    pos_x = pos_x.to(device)

    for t in range(1, L+1):
        for x in range(l, l+L):
            dep_set = torch.cat([phi_pad[:, t:t+1, x-l:x],
                                 phi_pad[:, t-1:t, x:x+l+1]],
                                dim=2)
            # dep_set.shape = (batch, 1, 2*l + 1)
            dep_flag = torch.cat([phi_flag[:, t:t+1, x-l:x],
                                 phi_flag[:, t-1:t, x:x+l+1]],
                                 dim=2)
            dep_t = torch.cat([pos_t[:, t:t+1, x-l:x],
                               pos_t[:, t-1:t, x:x+l+1]],
                              dim=2)
            dep_x = torch.cat([pos_x[:, t:t+1, x-l:x],
                               pos_x[:, t-1:t, x:x+l+1]],
                              dim=2)
            
            dep_set = torch.cat([dep_set, dep_flag,
                                 dep_t, dep_x], dim=1)
            sample, logp = net(dep_set, device)
            phi_pad[:, t, x] = sample
            phi_lp += logp
    phi = phi_pad[:, 1:L+1, l:l+L]
    return phi, phi_lp
# %%
def loss_reinforce(lattice: Lattice, phi, phi_lp):
    with torch.no_grad():
        loss = lattice.KL_loss(phi, phi_lp)
        loss = loss - loss.mean()

    return torch.mean(phi_lp*loss)

# %%
def train(lattice: Lattice, net: ConvNet, batch=100,
          epochs=100, lr=0.02, device='cpu'):
    optimizer = torch.optim.Adam(net.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 400, 0.5)

    for ep in range(epochs):
        # Zero your gradients for every batch
        optimizer.zero_grad()
        # Obtain samples and logp of given batch size
        phi, phi_lp = sample_fn(lattice, net, batch, device)
        # Compute loss and gradients
        l = loss_reinforce(lattice, phi, phi_lp)
        l.backward()
        # Adjust network parameters using optimizer and gradients
        optimizer.step()
        scheduler.step()

        if (ep+1) % 200 == 0:
            loss = lattice.KL_loss(phi, phi_lp)
            print('loss_mean: {}'.format(loss.mean()))
            print('loss_std: {}'.format(loss.std()))


# %%
device = 'cuda:0'
lattice = Lattice(30, 4.96)
net = ConvNet().to(device)
# %%
epochs = 400
batch = 64
train(lattice, net, batch, epochs, 0.01, device)
# %%
torch.save(net.state_dict(), 'ar_gauss_2d_chkpt.pth')
# %%
phi, phi_lp = sample_fn(lattice, net, batch, device)
loss = lattice.KL_loss(phi, phi_lp)
loss.mean(), loss.std()
# %%
