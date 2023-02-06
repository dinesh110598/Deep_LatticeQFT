# A coupling layer for RealNVP
struct RealNVPCoupling
    s::Chain
    t::Chain
end

function RealNVPCoupling(d1=3, d2=3, width=32)
    @argcheck depth1 > 1
    layers1 = [Dense(1 => width, relu)]
    append!(layers1, [Dense(width => width, relu)
                      for i=1:d1-2])
    append!(layers1, [Dense(width => 1)])
    s = Chain(layers1...)
    
    @argcheck depth2 > 1
    layers2 = [Dense(1 => width, relu)]
    append!(layers2, [Dense(width => width, relu)
                      for i=1:d2-2])
    append!(layers2, [Dense(width => 1)])
    t = Chain(layers2...)
    
    RealNVPCoupling(s, t)
end

function call(net::RealNVPCoupling, x, lp)
    x1, x2 = Flux.chunk(x, 2; dims=1)
    x1 = Flux.unsqueeze(x1; dims=1)
    
    s_x, t_x = net.s(x1)[1, :], net.t(x1)[1, :]
    sample = x2.*exp(s_x) .+ t_x
    logprob = lp - s_x
    return sample, logprob
end
