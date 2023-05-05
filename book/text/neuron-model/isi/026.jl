Random.seed!(0) # set random seed

T = 1000 # ms
dt = 1f0 # ms
nt = Int32(T/dt) # number of timesteps

n_neurons = 10 # ニューロンの数
fr = 30 # ガンマスパイクの発火率(Hz)

# k=1のときはポアソン過程に一致
spikes1 = GammaSpike(T, dt, n_neurons, fr, 1)
spikes2 = GammaSpike(T, dt, n_neurons, fr, 12)

println("Num. of spikes : ", sum(spikes1), ", ",sum(spikes2))
println("Firing rate : ", sum(spikes1)/(n_neurons*T)*1e3, "Hz, ", sum(spikes2)/(n_neurons*T)*1e3, "Hz")