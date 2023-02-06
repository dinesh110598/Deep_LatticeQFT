using Flux: randn32
using Flux, ArgCheck

struct Lattice
    L::Int
    a::AbstractFloat
    dim::Int
end

# Fix dim=2 (1 space + 1 time) for now
Lattice(L, a) = Lattice(L, a, 2)

struct DeepSets
    fc1::Chain
    fc2::Chain
    agg
end

function DeepSets(depth=3, width=32)
    @argcheck depth > 1
    layers1 = [Dense(2 => width, tanh)]
    append!(layers1, [Dense(width => width, relu) 
                      for i=1:depth-1])
    fc1 = Chain(layers1...) # NN of given depth and width
    
    layers2 = [Dense(width => width, relu)
               for i=1:depth-1]
    append!(layers2, [Dense(width => 2)])
    fc2 = Chain(layers2...) # NN of given depth and width
    
    agg = sum # aggregating function for DeepSets
    return DeepSets(fc1, fc2, agg)
end

"""
Applies deepsets network and outputs parameters of 
mixed Gaussian distribution
"""
function call(net::DeepSets, x) # size(x) = (3, 2, batch)
    x2 = net.fc1(x)
    # Aggregate along set dimension 2
    x2 = agg(x2; dims=2)[:, 1, :]
    # Obtain distribution params after acting 2nd net
    dist_param = net.fc2(x2)
    # Split between mean and variance
    μ, σ = Flux.chunk(dist_param, 2; dims=1)
    σ = exp.(logσ) # Make positive
    ϵ = randn32(size(μ)...)
    sample = (μ .+ ϵ.*σ)[1, :] # Gaussian sample, size: (batch,)
    logprob = (-log.(σ) - (ϵ^2)./2)[1, :]
end

function sample(lattice::Lattice, net::DeepSets, batch)
    L = lattice.L
    T = Float32
    # Lattice with 1 zero padding on every dim
    ϕ_pad = zeros(L+1, L+1, batch)
    # T store log probabilities of every sampled value
    ϕ_lp = zeros(L, L, batch)
    for t in 2:L+1, x in 2:L+1
        # Neighbour ϕ values at every lattice position
        ϕ_neigh = Flux.stack(
            [ϕ_pad[x-1, t, :], ϕ_pad[x, t-1, :]]; 
        dims=1)
        # Distance from zero for every dimension
        dist = Flux.stack(
            [fill(T((x-1)/L), batch), 
             fill(T((t-1)/L), batch)]; dims=1)
        # Flags if neighbour padded
        flag = Flux.stack(
            [fill(x==2 ? 1 : 0, batch),
             fill(t==2 ? 1 : 0, batch)]; dims=1)
        # Stacking them for NN input
        inp = Flux.stack([ϕ_neigh, dist, flag]; dims=1)
        # Get sample and logprob from NN
        sample, logprob = call(net, inp) #(M, batch) arrays
        ϕ_pad[x, t, :] = sample
        ϕ_lp[x-1, t-1, :] = logprob
    end
end