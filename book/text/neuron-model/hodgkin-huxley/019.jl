fig, axes = subplots(2,1, figsize=(4, 2), height_ratios=[1, 0.5], constrained_layout=true) 
axes[1].set_title("Connor-Stevens model")
axes[1].plot(time, varr_cs[:, 1], color="black"); axes[1].set_ylabel("V (mV)")
axes[2].plot(time, Iext_cs[:, 1], color="black"); 
axes[2].set_ylabel("Current\n"*L"($\mu$A/cm$^2$)"); axes[2].set_xlabel("Times (ms)")