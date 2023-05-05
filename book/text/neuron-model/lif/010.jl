figure(figsize=(4, 4))
subplot(2,1,1); plot(t, varr[:, 1]); ylabel("V (mV)")
subplot(2,1,2); plot(t, Ie[:, 1]); ylabel("Current");  xlabel("Times (ms)")
tight_layout()