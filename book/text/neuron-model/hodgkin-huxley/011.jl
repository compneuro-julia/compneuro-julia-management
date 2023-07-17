fig, axes = subplots(3,1, figsize=(5, 3), height_ratios=[1, 1, 0.5], constrained_layout=true) 
axes[1].set_title("Hodgkin-Huxley model")
axes[1].plot(time, varr[:, 1], color="black"); axes[1].set_ylabel("V (mV)")

labellist=["m" "h" "n"] 
for i in 1:3
    axes[2].plot(time, gatearr[:, i, 1], label=labellist[i])
end; 
axes[2].set_ylabel("Gating Value"); axes[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
axes[3].plot(time, Ie[:, 1], color="black"); 
axes[3].set_ylabel("Current\n"*L"($\mu$A/cm$^2$)"); axes[3].set_xlabel("Times (ms)")