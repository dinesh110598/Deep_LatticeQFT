# %%
import torch
from model_2d import Ising, ConvNet, train
# %%
betas = [0.44, 0.46]
ls = [9, 13]
filter_ws = [2, 4]
depths = [2, 4]
ep = 500

for b in betas:
    for l in ls:
        for fw in filter_ws:
            for depth in depths:
                lattice = Ising(30, b)
                net = ConvNet(l, fw, depth)
                print("beta = {}".format(b))
                train(lattice, net, 100, ep, 0.02, 'cuda:0')
                torch.save(net.state_dict(), 
                    'Saves/b{}_l{}_f{}_d{}.pth'.format(b, l, fw, depth))

# %%
