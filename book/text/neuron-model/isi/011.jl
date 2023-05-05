function rasterplot(spikes)
    # input spike -> time, #neurons
    spike_inds = Tuple.(findall(x -> x > 0, spikes))
    spike_time = first.(spike_inds)
    neuron_inds = last.(spike_inds)
    
    # raster plot
    scatter(spike_time, neuron_inds, s=1, c="black")
    xlabel("Time (ms)"); ylabel("Neuron index")
    tight_layout()
end