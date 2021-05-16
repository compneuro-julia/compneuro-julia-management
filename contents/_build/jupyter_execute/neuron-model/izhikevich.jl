using Base: @kwdef
using Parameters: @unpack # or using UnPack
using PyPlot

@kwdef struct IZParameter{FT}
    C::FT = 100  # 膜容量 (pF)
    a::FT = 0.03 # 回復時定数の逆数 (1/ms)
    b::FT = -2 # u の v に対する共鳴度合い (pA/mV)
    d::FT = 100 # 発火で活性化される正味の外向き電流 (pA)
    k::FT = 0.7 # ゲイン (pA/mV)
    vthr::FT = -40 # 閾値電位 (mV)
    vrest::FT = -60 # 静止膜電位 (mV)
    vreset::FT = -50 # リセット電位 (mV)
    vpeak::FT = 35 #　ピーク電位 (mV)
end

@kwdef mutable struct IZ{FT}
    param::IZParameter = IZParameter{FT}()
    N::UInt32
    v::Vector{FT} = fill(param.vrest, N)
    u::Vector{FT} = zeros(N)
    fire::Vector{Bool} = zeros(Bool, N)
end

function update!(variable::IZ, param::IZParameter, Ie::Vector, dt)
    @unpack N, v, u, fire = variable
    @unpack C, a, b, d, k, vthr, vrest, vreset, vpeak = param
    @inbounds for i = 1:N
        v[i] += dt/C * (k*(v[i]-vrest)*(v[i]-vthr) - u[i] + Ie[i])
        u[i] += dt * (a * (b * (v[i]-vrest) - u[i]))
    end
    @inbounds for i = 1:N
        fire[i] = v[i] >= vpeak
        v[i] = ifelse(fire[i], vreset, v[i])
        u[i] += ifelse(fire[i], d, 0)
    end
end;

T = 450 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 1 # ニューロンの数

# 入力刺激
t = Array{Float32}(1:nt)*dt
Ie = repeat(150f0 * ((t .> 50) - (t .> 200)) + 300f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# 記録用
varr, uarr = zeros(Float32, nt, N), zeros(Float32, nt, N)

# modelの定義
neurons = IZ{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, Ie[i, :], dt)
    varr[i, :], uarr[i, :] = neurons.v, neurons.u
end

figure(figsize=(4, 4))
suptitle("Regular Spiking (RS) Neurons")
subplot(3,1,1); plot(t, varr[:, 1]); ylabel("Membrane\n potential (mV)")
subplot(3,1,2); plot(t, uarr[:, 1]); ylabel("Recovery\n current (pA)")
subplot(3,1,3); plot(t, Ie[:, 1]); ylabel("Injection\n current (pA)"); xlabel("Times (ms)")
tight_layout(rect=[0,0,1,0.96])

# 記録用
varr_ib, varr_ch = zeros(Float32, nt, N), zeros(Float32, nt, N)
Ie = repeat(500f0 * ((t .> 50) - (t .> 200)) + 700f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# IB neurons
neurons_ib = IZ{Float32}(N=N, 
    param=IZParameter{Float32}(C = 150, a = 0.01, b = 5, k =1.2, d = 130, vrest = -75, vreset = -56, vthr = -45, vpeak = 50))

# CH neurons
neurons_ch = IZ{Float32}(N=N, 
    param=IZParameter{Float32}(C = 50, a = 0.03, b = 1, k =1.5, d = 150, vrest = -60, vreset = -40, vthr = -40, vpeak = 35))

# simulation
@time for i = 1:nt
    update!(neurons_ib, neurons_ib.param, Ie[i, :], dt)
    update!(neurons_ch, neurons_ch.param, Ie[i, :], dt)
    varr_ib[i, :], varr_ch[i, :] = neurons_ib.v, neurons_ch.v
end

figure(figsize=(6, 2))
subplot(1,2,1); plot(t, varr_ib[:, 1]); title("IB Neurons"); ylabel("Membrane\n potential (mV)");  xlabel("Times (ms)")
subplot(1,2,2); plot(t, varr_ch[:, 1]); title("CH neurons"); xlabel("Times (ms)")
tight_layout()

# Excitatory neurons, Inhibitory neurons
Ne, Ni = 800, 200;
re, ri = rand(Ne,1), rand(Ni,1)
a = [0.02ones(Ne,1); 0.02 .+ 0.08ri]
b = [0.2ones(Ne,1); 0.25 .- 0.05ri]
c = [-65 .+ 15re.^2; -65ones(Ni,1)]
d = [8 .- 6re.^2; 2ones(Ni,1)]
S = [0.5rand(Ne+Ni,Ne) -rand(Ne+Ni,Ni)] # synaptic weight
v = -65ones(Ne+Ni,1)   # Initial values of v
u = b .* v              # Initial values of u
firings = []            # spike timings

for t=1:1000 # simulation of 1000 ms
    Ie = [5randn(Ne,1); 2randn(Ni,1)] # thalamic input
    fired = findall(v[:, 1] .>= 30) # indices of spikes
    firings = t==1 ? [t .+ 0*fired fired] : [firings; [t .+ 0*fired fired]]
    v[fired] = c[fired]
    u[fired] += d[fired]
    Ie += sum(S[:,fired], dims=2)
    v += 0.5(0.04v.^2 + 5v .+140 - u + Ie) # step 0.5 ms for numerical stability
    v += 0.5(0.04v.^2 + 5v .+140 - u + Ie) 
    u += a .* (b .* v - u)
end

figure(figsize=(6, 3))
scatter(firings[:,1], firings[:,2], c="k", s=1, alpha=0.5)
xlabel("Time (ms)"); ylabel("# neuron"); xlim(0, 1000); ylim(0, 1000)
tight_layout()
