using Base: @kwdef
using Parameters: @unpack # or using UnPack

@kwdef struct HHParameter{FT}
    Cm::FT = 1.0 # 膜容量(uF/cm^2)
    gNa::FT = 120.0 # Na+ の最大コンダクタンス(mS/cm^2)
    gK::FT = 36.0 # K+ の最大コンダクタンス(mS/cm^2)
    gL::FT = 0.3 # 漏れイオンの最大コンダクタンス(mS/cm^2)
    ENa::FT = 50.0 # Na+ の平衡電位(mV)
    EK::FT = -77.0 # K+ の平衡電位(mV)
    EL::FT = -54.387 #漏れイオンの平衡電位(mV)
    tr::FT = 0.5 # ms
    td::FT = 8.0 # ms
    invtr::FT = 1.0 / tr
    invtd::FT = 1.0 / td
    v0::FT = -20.0 # mV
end

@kwdef mutable struct HH{FT}
    param::HHParameter = HHParameter{FT}()
    N::UInt16
    v::Vector{FT} = fill(-65.0, N)
    m::Vector{FT} = fill(0.05, N)
    h::Vector{FT} = fill(0.6, N)
    n::Vector{FT} = fill(0.32, N)
    r::Vector{FT} = zeros(N)
end

function update!(variable::HH, param::HHParameter, I::Vector, dt)
    @unpack N, v, m, h, n, r = variable
    @unpack Cm, gNa, gK, gL, ENa, EK, EL, tr, td, invtr, invtd, v0 = param
    @inbounds for i = 1:N
        m[i] += dt * ((0.1(v[i]+40.0)/(1.0 - exp(-0.1(v[i]+40.0))))*(1.0 - m[i]) - 4.0exp(-(v[i]+65.0) / 18.0)*m[i])
        h[i] += dt * ((0.07exp(-0.05(v[i]+65.0)))*(1.0 - h[i]) - 1.0/(1.0 + exp(-0.1(v[i]+35.0)))*h[i])
        n[i] += dt * ((0.01(v[i]+55.0)/(1.0 - exp(-0.1(v[i]+55.0))))*(1.0 - n[i]) - (0.125exp(-0.0125(v[i]+65)))*n[i])
        v[i] += dt / Cm * (I[i] - gNa * m[i]^3 * h[i] * (v[i] - ENa) - gK * n[i]^4 * (v[i] - EK) - gL * (v[i] - EL))
        r[i] += dt * ((invtr - invtd) * (1.0 - r[i])/(1.0 + exp(-v[i] + v0)) - r[i] * invtd)
    end
end

T = 450 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 1 # ニューロンの数

# 入力刺激
t = Array{Float32}(1:nt)*dt
I = repeat(10f0 * ((t .> 50) - (t .> 200)) + 35f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# 記録用
varr = zeros(Float32, nt, N)
gatearr = zeros(Float32, nt, 3, N)

# modelの定義
neurons = HH{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, I[i, :], dt)
    varr[i, :] = neurons.v
    gatearr[i, 1, :] = neurons.m
    gatearr[i, 2, :] = neurons.h
    gatearr[i, 3, :] = neurons.n
end

using Plots

p1 = plot(t, varr[:, 1], label=false, color="black")
p2 = plot(t, gatearr[:, :, 1], label=["m" "h" "n"])
p3 = plot(t, I[:, 1], label=false, color="black")
plot(p1, p2, p3, 
    xlabel = ["" "" "Times (ms)"], 
    ylabel= ["V (mV)" "Gating Value" "Current"],
    layout = grid(3, 1, heights=[0.4, 0.35, 0.25]), size=(600,500))

spike = (varr[1:nt-1, :] .< 0) .& (varr[2:nt, :] .> 0)
num_spikes = sum(spike, dims=1)
println("Num. of spikes : ", num_spikes[1])

T = 1000 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps

N = 100 # ニューロンの数

# 入力刺激
maxcurrent = 30
t = Array{Float32}(1:nt)*dt
I = Array{Float32}(range(1,maxcurrent,length=N)) # injection current

# modelの定義
neurons = HH{Float32}(N=N)

# 記録用
varr_fi = zeros(Float32, nt, N)

# simulation
for i = 1:nt
    update!(neurons, neurons.param, I[:], dt)
    varr_fi[i, :] = neurons.v
end

spike = (varr_fi[1:nt-1, :] .< 0) .& (varr_fi[2:nt, :] .> 0)
num_spikes = sum(spike, dims=1)
rate = num_spikes/T*1e3

plot(I[:], rate[1, :],
    xlabel="Input current",
    ylabel="Firing rate (Hz)", legend=false, size=(400,300))

T = 450 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 1 # ニューロンの数

# 入力刺激
t = Array{Float32}(1:nt)*dt
I = repeat(10f0 * (-(t .> 50) + (t .> 200)) + 20f0 * (-(t .> 250) + (t .> 400)), 1, N)  # injection current

# modelの定義
neurons = HH{Float32}(N=N)

# 記録用
varr2 = zeros(Float32, nt, N)
gatearr2 = zeros(Float32, nt, 3, N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, I[i, :], dt)
    varr2[i, :] = neurons.v
    gatearr2[i, 1, :] = neurons.m
    gatearr2[i, 2, :] = neurons.h
    gatearr2[i, 3, :] = neurons.n
end

p1 = plot(t, varr2[:, 1], label=false, color="black")
p2 = plot(t, gatearr2[:, :, 1], label=["m" "h" "n"])
p3 = plot(t, I[:, 1], label=false, color="black")
plot(p1, p2, p3, 
    xlabel = ["" "" "Times (ms)"], 
    ylabel= ["V (mV)" "Gating Value" "Injection\n current"],
    layout = grid(3, 1, heights=[0.4, 0.35, 0.25]), size=(600,500))
