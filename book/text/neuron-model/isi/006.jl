figure(figsize=(6, 2))
for i=1:n_neurons
    scatter(spike_time[:, i], i*ones(Int(nt*1.5/fr)), c="k", s=1)
end
xlabel("Time (ms)"); ylabel("Neuron index"); xlim(0, T+10); tight_layout()