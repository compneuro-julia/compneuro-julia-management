T = 1000 # ms
dt = 0.05 # ms
nt = Int(T/dt) # number of timesteps

N = 100 # ニューロンの数

# 入力刺激
mincurrent, maxcurrent = 1, 30
t = (1:nt)*dt
Ie_range = Array{Float32}(range(mincurrent, maxcurrent, length=N)) # injection current

# modelの定義
neurons = HH{Float32}(N=N)

# 記録用
varr_fi = zeros(Float32, nt, N)

# simulation
for i = 1:nt
    update!(neurons, neurons.param, Ie_range, dt)
    varr_fi[i, :] = neurons.v
end