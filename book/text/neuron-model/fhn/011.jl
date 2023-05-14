figure(figsize=(5,4))
subplot(2, 1, 1); plot(t, varr[:, 1], label=false, color="black"); ylabel("v")
subplot(2, 1, 2); plot(t, uarr[:, 1], label=false); ylabel("u"); xlabel("Times (ms)")
tight_layout()