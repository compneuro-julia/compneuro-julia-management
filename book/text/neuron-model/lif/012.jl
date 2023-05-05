T = 1000 # ms
dt = 0.01f0 # ms
nt = UInt32(T/dt) # number of timesteps
N = 100 # ニューロンの数

# 入力刺激
mincurrent, maxcurrent = 15, 40
t = Array{Float32}(1:nt)*dt
Ie = Array{Float32}(range(mincurrent,maxcurrent,length=N)) # injection current

# modelの定義
neurons = LIF{Float32}(N=N)

# 記録用
firearr = zeros(Bool, nt, N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, Ie, dt)
    neurons.tcount += 1
    firearr[i, :] = neurons.fire
end