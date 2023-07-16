T = 50 # ms
dt = 0.01f0 # ms
nt = Int32(T/dt) # number of timesteps
num_neurons = 1 # ニューロンの数

# 入力刺激
time = Array{Float32}(1:nt)*dt
Ie = repeat(5*((time .> 10) - (time .> 15)), 1, num_neurons)  # injection current

# 記録用
varr, rarr = zeros(Float32, nt, num_neurons), zeros(Float32, nt, num_neurons)

# modelの定義
hh_neurons = HH{Float32}(num_neurons=num_neurons, dt=dt)
hh_synapse = HHKineticSynapse{Float32}(num_neurons=num_neurons, dt=dt)

# simulation
@time for t = 1:nt
    v = hh_neurons(Ie[t, :])
    r = hh_synapse(v)
    varr[t, :], rarr[t, :] = v, r
end