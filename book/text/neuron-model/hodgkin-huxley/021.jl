figure(figsize=(4, 3))
plot(Ie_range[:], rate[1, :]); xlabel("Input current"); ylabel("Firing rate (Hz)")
tight_layout()