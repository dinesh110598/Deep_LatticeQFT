# %%
import torch
from ar_flow import Lattice, DeepSets
from ar_flow import sample, S_scalar, train
from time import time
# %%
depths = [5]
d2 = 5
nlayers = 6
L = 24
batch = 100
device='cuda:0'
epochs = 1500
lr = 0.01
s_step = 300


for depth in depths:
    lattice = Lattice(L)
    net = DeepSets(depth, 32, 'sum', nlayers,
                    d2, device).to(device)
    start_t = time()
    train(lattice, net, batch, epochs, lr, s_step, device)
    end_t = time()
    phi, phi_lp = sample(net, lattice, batch, device)
    S = S_scalar(lattice, phi, device)
    loss = S + phi_lp
    
    print('depth = {}'.format(depth))
    print('Training time(s) = {}'.format(end_t-start_t))
    print('Mean loss = {}'.format(loss.mean()))
    print('Loss std = {}'.format(loss.std()))
    print('================')
    torch.save(net.state_dict(), 
                'Saves/d{}_d2{}'.format(depth, d2))

# %%
