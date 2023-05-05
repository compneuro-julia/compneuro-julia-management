figure(figsize=(6, 3))
scatter(firings[:,1], firings[:,2], c="k", s=1, alpha=0.5)
xlabel("Time (ms)"); ylabel("# neuron"); xlim(0, 1000); ylim(0, 1000)
tight_layout()