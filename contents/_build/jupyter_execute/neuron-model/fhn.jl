using Base: @kwdef
using Parameters: @unpack # or using UnPack
using PyPlot, PyCall
rcParams = PyDict(plt."rcParams")
rcParams["axes.spines.top"], rcParams["axes.spines.right"] = false, false;

@kwdef struct FHNParameter{FT}
    a::FT = 0.7
    b::FT = 0.8
    c::FT = 10.0
end

@kwdef mutable struct FHN{FT}
    param::FHNParameter = FHNParameter{FT}()
    N::UInt16
    v::Vector{FT} = fill(-1.0, N)
    u::Vector{FT} = zeros(N)
end

function update!(variable::FHN, param::FHNParameter, Ie::Vector, dt)
    @unpack N, v, u = variable
    @unpack a, b, c = param
    @inbounds for i = 1:N
        v[i] += dt * c * (-u[i] + v[i] - v[i]^3 / 3 + Ie[i])
        u[i] += dt * (v[i] - b*u[i] + a)
    end
end

T = 50 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 1 # ニューロンの数

# 入力刺激
t = Array{Float32}(1:nt)*dt
Ie = repeat(0.35f0*ones(nt), 1, N)  # injection current

# 記録用
varr, uarr = zeros(Float32, nt, N), zeros(Float32, nt, N)

# modelの定義
neurons = FHN{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, Ie[i, :], dt)
    varr[i, :], uarr[i, :] = neurons.v, neurons.u
end

figure(figsize=(5,4))
subplot(2, 1, 1); plot(t, varr[:, 1], label=false, color="black"); ylabel("v")
subplot(2, 1, 2); plot(t, uarr[:, 1], label=false); ylabel("u"); xlabel("Times (ms)")
tight_layout()

spike = (varr[1:nt-1, :] .< 0) .& (varr[2:nt, :] .> 0)
num_spikes = sum(spike, dims=1)
println("Num. of spikes : ", num_spikes[1]);

margin = 1.0
vmax, vmin = maximum(varr) + margin, minimum(varr) - margin
umax, umin = maximum(uarr) + margin, minimum(uarr) - margin
vrange, urange = vmin:0.1:vmax, umin:0.1:umax
U = [i for i in urange, j in 1:length(vrange)]
V = [j for i in 1:length(urange), j in vrange]

a, b, c, Ie = 0.7, 0.8, 10.0, 0.34
dV = c * (-U + V - V .^3 / 3 .+ Ie)
dU = V - b*U .+ a;

figure(figsize=(4,3))
streamplot(V, U, dV, dU, density=[0.8, 0.8], linewidth=2) 
contour(V, U, dU, levels=[0])
contour(V, U, dV, levels=[0])
    plot(varr, uarr); xlim(vmin, vmax); ylim(umin, umax); xlabel(L"$v$"); ylabel(L"$u$")
tight_layout()
