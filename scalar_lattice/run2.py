# %%
from ar_flow_2d import *
# %%
def sample_symmetric(lattice: Lattice, net: ConvNet, batch=100,
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
    
    return phi, phi_lp/6, S_scalar

def metropolis(lattice: Lattice, net: ConvNet, batch=1000,
               steps=10000, device='cpu'):
    epochs = steps//batch
    acc = 0
    curr_x = None
    curr_model_lp = None
    curr_target_lp = None

    for ep in range(epochs):
        with torch.no_grad():
            phi, phi_lp, S = sample_symmetric(lattice, net,
                                              batch, device)
            # phi, phi_lp = sample_fn(lattice, net, batch, device)
            # S = lattice.S(phi)
        
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
            acc_prob = torch.exp(curr_model_lp - prop_model_lp +
                                 prop_target_lp - curr_target_lp)

            if np.random.rand() < acc_prob:
                curr_x = prop_x
                curr_model_lp = prop_model_lp
                curr_target_lp = prop_target_lp
                acc += 1
    return acc
# %%
lattice = Lattice(16, 5.05)
device = 'cuda:0'
net = ConvNet(64).to(device)

net.load_state_dict(torch.load('Saves/ar_flow_2d_chkpt.pth'))
# %%
acc = metropolis(lattice, net, 400, 10000, device)
print(acc)
# %%
