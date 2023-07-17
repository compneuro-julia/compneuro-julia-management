function fi_curve(NeuronType; num_neurons=200, T=1000, dt=0.04,
                  current_range = [1, 30])
    nt = Int(T/dt) # number of timesteps
    Iext_range = Array{Float32}(range(current_range..., length=num_neurons)) # injection current
    neurons = NeuronType{Float32}(num_neurons=num_neurons, dt=dt) # modelの定義
    varr_fi = zeros(Float32, nt, num_neurons) # 記録用

    # simulation
    for t = 1:nt
        v = neurons(Iext_range)
        varr_fi[t, :] = v
    end
    num_spikes = get_num_spikes(varr_fi)
    rate = num_spikes/T*1e3;
    threshold = Iext_range[findfirst(rate .> 1)[2]]
    return Iext_range, rate, threshold
end