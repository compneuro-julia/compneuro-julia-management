figure(figsize=(5, 2.5))
subplot(1,2,1)
title("Type I (CS model)")
text(threshold_cs+1, 0, L"$I_{\theta}$="*string(round(threshold_cs, digits=2)))
plot(Iext_range_cs[:], rate_cs[1, :]); xlabel(L"Input current ($\mu$A/cm$^2$)"); ylabel("Firing rate (Hz)")

subplot(1,2,2)
title("Type II (HH model)")
text(threshold_hh+1, 0, L"$I_{\theta}$="*string(round(threshold_hh, digits=2)))
plot(Iext_range_hh[:], rate_hh[1, :]); xlabel(L"Input current ($\mu$A/cm$^2$)"); 
tight_layout()