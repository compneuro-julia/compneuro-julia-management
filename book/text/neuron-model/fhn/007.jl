T = 100 # ms
dt = 0.01 # ms
nt = UInt32(T/dt) # number of timesteps
num_neurons = 1 # ニューロンの数

# 入力刺激
time = (1:nt)*dt
Ie = repeat(0.5 * ((time .> 10) - (time .> 45)) + 0.34 * ((time .> 55) - (time .> 90)), 1, num_neurons)  # injection current

# 記録用
varr, uarr = zeros(Float32, nt, num_neurons), zeros(Float32, nt, num_neurons)

# modelの定義
fhn_neurons = FHN{Float32}(num_neurons=num_neurons, dt=dt)

# simulation
@time for t = 1:nt
    v = fhn_neurons(Ie[t, :])
    varr[t, :], uarr[t, :] = v, fhn_neurons.u
end