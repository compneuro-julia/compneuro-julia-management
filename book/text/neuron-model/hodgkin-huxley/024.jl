fig, axes = subplots(1,2, figsize=(4, 2),constrained_layout=true) 
axes[1].set_title("Type I (CS model)")
axes[1].text(threshold_cs+1, 0, L"$I_{\theta}$="*string(round(threshold_cs, digits=2)))
axes[1].plot(Iext_range_cs[:], rate_cs[1, :]); 
axes[1].set_ylabel("Firing rate (Hz)")

axes[2].set_title("Type II (HH model)")
axes[2].text(threshold_hh+1, 0, L"$I_{\theta}$="*string(round(threshold_hh, digits=2)))
axes[2].plot(Iext_range_hh[:], rate_hh[1, :]); 
fig.supxlabel(L"Input current ($\mu$A/cm$^2$)", size=10)