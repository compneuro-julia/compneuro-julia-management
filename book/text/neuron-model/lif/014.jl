num_spikes = sum(firearr, dims=1)
rate_numeric = num_spikes/T*1e3; 