T = 450 # ms
dt = 0.01 # ms
nt = Int(T/dt) # number of timesteps
N = 1 # ニューロンの数

# 入力刺激
t = Array{Float32}(1:nt)*dt
Ie = repeat(10f0 * (-(t .> 50) + (t .> 200)) + 20f0 * (-(t .> 250) + (t .> 400)), 1, N)  # injection current

# modelの定義
neurons = HH{Float32}(N=N)

# 記録用
varr2, gatearr2 = zeros(nt, N), zeros(nt, 3, N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, Ie[i, :], dt)
    varr2[i, :] = neurons.v
    gatearr2[i, :, :] .= [neurons.m; neurons.h; neurons.n]
end