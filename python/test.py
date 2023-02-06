# %%
import torch
from torch import nn
# %%
L = 50
a = 1
phi = torch.randn(L, L)
# %%
x = phi[0:2, 0:2]
y = torch.cat((x, x), dim=1)
phi[0, 0] = -0.9
y[0, 0]
# %%
batch_size = 20
# %%