##
a = 1
L = 50
##
ϕ = randn(Float32, L, L) # 1D space + time
dist = zeros(Float32, L, L, 2)
for i in 1:L, j in 1:L
    dist[i, j, 1] = (1/L)*i
    dist[i, j, 2] = (1/L)*j
end
##

##