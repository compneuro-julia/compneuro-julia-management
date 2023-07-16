figure(figsize=(5,4))
subplot(3, 1, 1); plot(time, varr[:, 1], label=false, color="black"); ylabel("v")
subplot(3, 1, 2); plot(time, uarr[:, 1], label=false); ylabel("u"); 
subplot(3, 1, 3); plot(time, Ie, label=false); ylabel("Current"); xlabel("Times (ms)")
tight_layout()