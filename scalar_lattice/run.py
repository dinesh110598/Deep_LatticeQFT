# %%
import torch
from ar_flow_2d import *
from time import time
# %%
lattice = Lattice(16, 5.05)
device = 'cuda:0'
net = ConvNet(64).to(device)
# %%
epochs = 8000
batch = 200
train(lattice, net, batch, epochs, 1e-2, 800, device)
# %%
torch.save(net.state_dict(), 'Saves/ar_flow_2d_chkpt.pth')
# %%
