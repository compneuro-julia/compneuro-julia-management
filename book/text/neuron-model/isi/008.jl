spike_time[spike_time .> nt - 1] .= 1 # ntを超える場合を1に
spike_time = round.(Int, spike_time) # float to int
spikes = zeros(nt, n_neurons) # スパイク記録変数

for i=1:n_neurons    
    spikes[spike_time[:, i], i] .= 1
end

spikes[1] = 0 # (spike_time=1)の発火を削除
println("Num. of spikes : ", sum(spikes))
println("Firing rate : ", sum(spikes)/(n_neurons*T)*1e3, "Hz")